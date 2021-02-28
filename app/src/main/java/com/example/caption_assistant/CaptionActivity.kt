package com.example.caption_assistant

import android.database.sqlite.SQLiteDatabase
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.ThumbnailUtils
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
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
import java.io.File
import java.io.FileOutputStream
import kotlin.coroutines.CoroutineContext
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.round


data class ClassStatistic(
        val label: Map<Long, String>,
        val mean: INDArray,
        val std: INDArray,
        val counts: INDArray
)

class CaptionActivity : AppCompatActivity(), CoroutineScope {

    private val topK: Int = 5

    private var imageUri: Uri? = null
    private val imageWidth: Int = 256
    private val imageHeight: Int = 256
    private val embeddingSize: Int = 512

    private lateinit var imageView: ImageView
    private lateinit var progressBar: ProgressBar
    private lateinit var progressBarText: TextView
    private lateinit var predictionsChipGroup: ChipGroup
    private lateinit var confirmButton: Button

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
        get() = Dispatchers.Main + job

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        job = Job()

        setContentView(R.layout.activity_caption)
        imageView = findViewById(R.id.chosen_image)
        progressBar = findViewById(R.id.progress_bar)
        progressBarText = findViewById(R.id.progress_bar_text)
        predictionsChipGroup = findViewById(R.id.predicted_classes)
        confirmButton = findViewById(R.id.confirm_button)

        progressBar.visibility = View.INVISIBLE
        progressBarText.visibility = View.INVISIBLE
        confirmButton.visibility = View.INVISIBLE

        confirmButton.setOnClickListener { this.finish() }
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
                progressBar.visibility = View.VISIBLE
                progressBarText.visibility = View.VISIBLE

                launch {
                    progressBarText.text = "Inferring tags..."

                    val predictedLabels = runInference(bitmap).toMutableList()

                    progressBarText.visibility = View.INVISIBLE
                    progressBar.visibility = View.INVISIBLE
                    confirmButton.visibility = View.VISIBLE

                    predictedLabels.forEach { label ->
                        val chip = Chip(predictionsChipGroup.context)
                        chip.text = label
                        chip.isCloseIconVisible = true

                        chip.setOnCloseIconClickListener {
                            predictedLabels.remove(label)
                            predictionsChipGroup.removeView(it)
                        }
                        predictionsChipGroup.addView(chip)
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

    private suspend fun predictGaussianNaiveBayes(embedding: INDArray, topK: Int = this.topK): List<Long> {
        return withContext(Dispatchers.Default) {
            val frequencies = sampleCounts.div(sampleCounts.sumNumber())
            val priors = Transforms.log(frequencies.add(exp(-120.0)))

            launch(Dispatchers.Main) { progressBar.isIndeterminate = false }

            val jointLogLikelihood = Nd4j.zeros(numClasses)
            for (cid in 0 until numClasses.toLong()) {
                val prior = priors.getDouble(cid)
                val mean = classMean.getRow(cid).castTo(DataType.DOUBLE)
                val std = classStd.getRow(cid).castTo(DataType.DOUBLE)

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
                jointLogLikelihood.putScalar(cid, prior * posterior)

                launch(Dispatchers.Main) {
                    progressBar.progress = round(100 * cid.toFloat() / numClasses.toFloat()).toInt()
                }
            }

            launch(Dispatchers.Main) { progressBar.isIndeterminate = true }

            IntRange(1, min(topK, numClasses)).map {
                val predCid = jointLogLikelihood.argMax().getLong(0)
                jointLogLikelihood.putScalar(predCid, -1e9)
                predCid
            }.toList()
        }
    }

    private suspend fun runInference(bitmap: Bitmap): List<String> {
        return withContext(Dispatchers.Default) {
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

            val (classLabels_, classMean_, classStd_, sampleCounts_) = statsLoading.await()
            classLabels = classLabels_
            classMean = classMean_
            classStd = classStd_
            sampleCounts = sampleCounts_

            val predictionIds = predictGaussianNaiveBayes(embedding)

            predictionIds.map { pred -> classLabels[pred]!! }
        }
    }
}
