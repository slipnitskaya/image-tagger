package com.example.caption_assistant

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.ThumbnailUtils
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.database.sqlite.transaction
import androidx.core.view.get
import androidx.core.view.size
import com.daveanthonythomas.moshipack.MoshiPack
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import kotlinx.coroutines.*
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import kotlin.coroutines.CoroutineContext
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.min


data class ClassStatistic(
    val label: Map<Long, String>,
    val mean: INDArray,
    val std: INDArray,
    val counts: INDArray
)

class CaptionActivity : AppCompatActivity(), CoroutineScope {

    private val momentum: Float = 0.05f

    private var imageUri: Uri? = null
    private val imageWidth: Int = 256
    private val imageHeight: Int = 256
    private val embeddingSize: Int = 512

    private val predictions = HashMap<String, List<Long>>()

    private lateinit var imageView: ImageView
    private lateinit var progressBar: ProgressBar
    private lateinit var progressBarText: TextView
    private lateinit var discardButton: Button
    private lateinit var addTagButton: Button
    private lateinit var confirmButton: Button
    lateinit var predictionsChipGroup: ChipGroup

    private lateinit var classMean: INDArray
    private lateinit var classStd: INDArray
    private lateinit var sampleCounts: INDArray
    private lateinit var classLabels: Map<Long, String>
    private lateinit var model: Module

    private val numClasses: Int
        get() = classLabels.size

    private val previewSize: Pair<Int, Int>
        get() = Pair(imageWidth, imageHeight)

    private lateinit var statsLoading: Deferred<ClassStatistic>
    private lateinit var modelLoading: Deferred<Module>
    private lateinit var job: Job
    override val coroutineContext: CoroutineContext
        get() = Dispatchers.Default + job

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        job = Job()

        setContentView(R.layout.activity_caption)
        imageView = findViewById(R.id.chosen_image)
        progressBar = findViewById(R.id.progress_bar)
        progressBarText = findViewById(R.id.progress_bar_text)
        predictionsChipGroup = findViewById(R.id.predicted_classes)
        discardButton = findViewById(R.id.discard_button)
        addTagButton = findViewById(R.id.add_button)
        confirmButton = findViewById(R.id.confirm_button)

        progressBar.visibility = View.INVISIBLE
        progressBarText.visibility = View.INVISIBLE
        discardButton.visibility = View.INVISIBLE
        addTagButton.visibility = View.INVISIBLE
        confirmButton.visibility = View.INVISIBLE

        discardButton.setOnClickListener { this.finish() }

        addTagButton.setOnClickListener { btn ->
            btn as Button

            with(TagListDialog(this, classLabels)) {
                show()
            }
            getSystemService(Context.INPUT_METHOD_SERVICE).also { imm ->
                imm as InputMethodManager
                imm.toggleSoftInput(InputMethodManager.SHOW_FORCED, InputMethodManager.HIDE_IMPLICIT_ONLY)
            }
        }

        // update statistics
        confirmButton.setOnClickListener {
            val embedding = imageView.tag as INDArray

            val contentValues = HashMap<Long, ContentValues>()
            val msgPack = MoshiPack()
            IntRange(0, predictionsChipGroup.childCount - 1)
                    .map { idx -> predictionsChipGroup[idx].id.toLong() }
                    .forEach { cid ->
                        val mean = classMean.getRow(cid).castTo(DataType.DOUBLE)
                        val std = classStd.getRow(cid).castTo(DataType.DOUBLE)
                        val count = sampleCounts.getInt(cid.toInt())

                        val diff = embedding.sub(mean)
                        val inc = diff.mul(momentum)
                        val newMean = mean.add(inc)
                        val newStd = std.add(diff.mul(inc)).mul(1.0 - momentum)
                        val newCount = count + 1

                        classMean.putRow(cid, newMean.castTo(classMean.dataType()))
                        classStd.putRow(cid, newStd.castTo(classStd.dataType()))
                        sampleCounts.putScalar(cid, newCount)

                        val cv = ContentValues()
                        cv.put("mean", msgPack.packToByteArray<FloatArray>(newMean.toFloatVector()))
                        cv.put("std", msgPack.packToByteArray<FloatArray>(newStd.toFloatVector()))
                        cv.put("counts", newCount)
                        contentValues[cid] = cv
                    }

            val tempFile = getTempFileOrCopy("caption_assistant.db")
            SQLiteDatabase
                    .openDatabase(tempFile, SQLiteDatabase.OpenParams.Builder().build())
                    .use { dbConn ->
                        dbConn.transaction {
                            contentValues.forEach { (cid, cv) ->
                                dbConn.update(
                                        "statistics",
                                        cv,
                                        "cid = ?",
                                        arrayOf(cid.toString())
                                )
                            }
                        }

                    }

            this.finish()
        }

        statsLoading = async(Dispatchers.IO) { loadStatisticsFromDB() }
        modelLoading = async(Dispatchers.IO) { loadModel() }
    }

    override fun onResume() {
        super.onResume()

        imageUri = intent.extras?.get("ImageURI") as Uri?
        if (imageUri != null) {
            val bitmap = loadBitmap(imageUri!!)
            if (bitmap != null) {
                imageView.setImageBitmap(bitmap)

                val md5 = ByteArrayOutputStream().use { baos ->
                    MessageDigest
                        .getInstance("MD5")
                        .apply {
                            update(baos.apply {
                                bitmap.compress(Bitmap.CompressFormat.PNG, 100, this)
                            }.toByteArray())
                        }
                        .digest()
                        .map { hexByte -> hexByte.toInt().and(0xFF).toString(16) }
                        .reduce { acc, s -> "${acc}${s}" }
                }

                launch(coroutineContext) {
                    val predictedClassIds = when(predictions.containsKey(md5)) {
                        true -> predictions[md5]
                        false -> {
                            launch(Dispatchers.Main) {
                                progressBar.visibility = View.VISIBLE
                                progressBarText.visibility = View.VISIBLE
                                progressBarText.text = "Inferring tags..."
                            }
                            runInference(bitmap).also { predictions[md5] = it }
                        }
                    }

                    launch(Dispatchers.Main) {
                        progressBar.visibility = View.INVISIBLE
                        progressBarText.visibility = View.INVISIBLE
                        discardButton.visibility = View.VISIBLE
                        addTagButton.visibility = View.VISIBLE
                        confirmButton.visibility = View.VISIBLE

                        val predictedLabels = predictedClassIds
                                ?.map { cid -> classLabels[cid] }
                                ?.toMutableList()
                                ?: mutableListOf()

                        predictedLabels.forEachIndexed { idx, label ->
                            val chip = Chip(predictionsChipGroup.context)
                            chip.id = predictedClassIds!![idx].toInt()
                            chip.text = label
                            chip.isCloseIconVisible = true

                            chip.setOnCloseIconClickListener {
                                predictedLabels.remove(label)
                                predictionsChipGroup.removeView(it)
                                if (predictionsChipGroup.size > 0) {
                                    enableConfirmButton()
                                } else {
                                    disableConfirmButton()
                                }
                            }
                            predictionsChipGroup.addView(chip)
                            enableConfirmButton()
                        }
                    }
                }
            }
        }
    }

    override fun onPause() {
        super.onPause()

        job.cancel()
    }

    private fun calculateInSampleSize(options: BitmapFactory.Options, reqWidth: Int, reqHeight: Int): Int {
        // Raw height and width of image
        val (height: Int, width: Int) = options.run { outHeight to outWidth }
        var inSampleSize = 1

        if (height > reqHeight || width > reqWidth) {

            val halfHeight: Int = height / 2
            val halfWidth: Int = width / 2

            // Calculate the largest inSampleSize value that is a power of 2 and keeps both
            // height and width larger than the requested height and width.
            while (halfHeight / inSampleSize >= reqHeight && halfWidth / inSampleSize >= reqWidth) {
                inSampleSize *= 2
            }
        }

        return inSampleSize
    }

    private fun loadBitmap(imageUri: Uri): Bitmap? {
        val (reqWidth, reqHeight) = previewSize

        return BitmapFactory.Options().run {
            inJustDecodeBounds = true

            contentResolver.openInputStream(imageUri).use {
                BitmapFactory.decodeStream(it, null, this)
            }

            inSampleSize = calculateInSampleSize(this, reqWidth, reqHeight)
            inJustDecodeBounds = false

            contentResolver.openInputStream(imageUri).use {
                BitmapFactory.decodeStream(it, null, this)
            }
        }
    }

    private fun getTempFileOrCopy(filename: String): File {
        val tempDir = externalCacheDir ?: cacheDir
        val tempFile = File(tempDir, filename)
        if (!tempFile.exists() && tempFile.createNewFile()) {
            FileOutputStream(tempFile).use { fos ->
                assets.open(filename).use { fis ->
                    fis.copyTo(fos)
                }
            }
        }

        return tempFile
    }

    private fun loadStatisticsFromDB(): ClassStatistic {
        val tempFile = getTempFileOrCopy("caption_assistant.db")

        val classLabels = HashMap<Long, String>()
        var classMean: INDArray?
        var classStd: INDArray?
        val sampleCounts: INDArray?

        SQLiteDatabase
                .openDatabase(tempFile, SQLiteDatabase.OpenParams.Builder().build())
                .use { dbConn ->
                    dbConn.query(
                        "statistics",
                        arrayOf("cid", "label", "mean", "std", "counts"),
                        null,
                        null,
                        null,
                        null,
                        null
                    ).use { cur ->
                        val numClasses = cur.count

                        classMean = Nd4j.zeros(numClasses, embeddingSize).castTo(DataType.FLOAT)
                        classStd = Nd4j.zeros(numClasses, embeddingSize).castTo(DataType.FLOAT)
                        sampleCounts = Nd4j.zeros(numClasses).castTo(DataType.FLOAT)

                        // instantiate de-serializer from BLOB to INDArray
                        val msgPack = MoshiPack()

                        generateSequence { if (cur.moveToNext()) cur else null }
                            .forEach { row ->
                                val classId = row.getLong(row.getColumnIndex("cid"))
                                val label = row.getString(row.getColumnIndex("label"))
                                val mean = Nd4j.create(msgPack.unpack<FloatArray>(
                                        row.getBlob(row.getColumnIndex("mean"))
                                ))
                                val std = Nd4j.create(msgPack.unpack<FloatArray>(
                                        row.getBlob(row.getColumnIndex("std"))
                                ))
                                val count = row.getInt(row.getColumnIndex("counts"))

                                classMean!!.putRow(classId, mean)
                                classStd!!.putRow(classId, std)
                                classLabels[classId] = label
                                sampleCounts!!.putScalar(classId, count)
                            }
                    }
                }

        return ClassStatistic(classLabels, classMean!!, classStd!!, sampleCounts!!)
    }

    private fun loadModel(): Module {
        val tempFile = getTempFileOrCopy("model.pt")
        return Module.load(tempFile.absolutePath)
    }

    private suspend fun calculateGaussianLikelihood(
        embedding: INDArray, prior: Double, mean: INDArray, std: INDArray
    ): Double {
        return withContext(coroutineContext) {
            val x = embedding.sub(mean)
            val posterior = x
                .mul(x)
                .div(std)
                .sum()
                .add(
                    Transforms.log(
                        std.mul(2.0 * PI)
                    ).sumNumber()
                ).sumNumber().toDouble()

            prior * posterior
        }
    }

    private suspend fun predictGaussianNaiveBayes(embedding: INDArray): List<Long> {
        return withContext(coroutineContext) {
            val frequencies = sampleCounts.div(sampleCounts.sumNumber())
            val priors = Transforms.log(frequencies.add(exp(-120.0)))

            launch(Dispatchers.Main) {
                progressBar.isIndeterminate = false
                progressBar.min = 0
                progressBar.max = numClasses
                progressBar.progress = 0
            }

            val jobs = ArrayList<Deferred<Job>>()
            val jointLogLikelihood = Nd4j.zeros(numClasses)
            for (cid in 0 until numClasses.toLong()) {
                val prior = priors.getDouble(cid)
                val mean = classMean.getRow(cid).castTo(DataType.DOUBLE)
                val std = classStd.getRow(cid).castTo(DataType.DOUBLE)

                val j = async(coroutineContext) {
                    val jLL = calculateGaussianLikelihood(embedding, prior, mean, std)
                    jointLogLikelihood.putScalar(cid, jLL)

                    launch(Dispatchers.Main) {
                        progressBar.progress += 1
                    }
                }
                jobs.add(j)
            }
            jobs.awaitAll()

            launch(Dispatchers.Main) { progressBar.isIndeterminate = true }

            launch(Dispatchers.Main) { progressBar.isIndeterminate = true }

            // predict top labels
            val predictedClassIds = mutableListOf<Long>()
            val probabilities = Transforms.softmax(
                    jointLogLikelihood.div(
                            Transforms.abs(jointLogLikelihood).maxNumber()
                    ).mul(50)
            )
            var cumProbability = 0.0
            while (cumProbability < 0.9) {
                val predictedCid = probabilities.argMax().getLong(0)
                predictedClassIds.add(predictedCid)
                cumProbability += probabilities.getDouble(predictedCid)
                probabilities.putScalar(predictedCid, 0.0)
            }
            predictedClassIds
        }
    }

    private suspend fun runInference(bitmap: Bitmap): List<Long> {
        return withContext(coroutineContext) {
            val shortSide = min(bitmap.height, bitmap.width)
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                Bitmap.createScaledBitmap(
                    ThumbnailUtils.extractThumbnail(bitmap, shortSide, shortSide),
                    imageWidth,
                    imageHeight,
                    true
                ),
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
            )

            model = modelLoading.await()

            val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()
            val output = outputTensor.dataAsFloatArray
            val embedding = Nd4j.createFromArray(*output).getRow(0)

            launch(Dispatchers.Main) { imageView.tag = embedding }

            val (classLabels_, classMean_, classStd_, sampleCounts_) = statsLoading.await()
            classLabels = classLabels_
            classMean = classMean_
            classStd = classStd_
            sampleCounts = sampleCounts_

            predictGaussianNaiveBayes(embedding)
        }
    }

    fun enableConfirmButton() {
        confirmButton.isEnabled = true
    }

    fun disableConfirmButton() {
        confirmButton.isEnabled = false
    }
}
