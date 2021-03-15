package com.example.image_tagger

import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.children
import androidx.core.view.forEach
import androidx.core.view.get
import androidx.core.view.size
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import kotlinx.coroutines.*
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import kotlin.coroutines.CoroutineContext

class TaggerActivity : AppCompatActivity(), CoroutineScope by MainScope() {

    private var imageUri: Uri? = null
    private var inferenceIsRunning = false
    private val predictions = HashMap<String, List<Int>>()

    private lateinit var imageView: ImageView
    private lateinit var progressBar: ProgressBar
    private lateinit var cancelButton: Button
    private lateinit var addTagButton: Button
    private lateinit var confirmButton: Button
    private lateinit var predictionsChipGroup: ChipGroup

    private val statsTable by lazy { ImageTagger.db.statisticsDao() }
    private val gnb by lazy { GaussianNaiveBayes(this) }

    private lateinit var job: Job
    override val coroutineContext: CoroutineContext
        get() = Dispatchers.Main + job

    lateinit var theTags: Map<Int, String?>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_tagger)

        imageView = findViewById(R.id.chosen_image)
        progressBar = findViewById(R.id.progress_bar)
        predictionsChipGroup = findViewById(R.id.predicted_classes)
        cancelButton = findViewById(R.id.discard_button)
        addTagButton = findViewById(R.id.add_button)
        confirmButton = findViewById(R.id.confirm_button)

        job = Job()
        launch {
            theTags = withContext(Dispatchers.IO) {
                statsTable.getStatistics()
                        .map { it.cid to it.label }
                        .toMap()
            }
        }

        cancelButton.setOnClickListener { performCancelAction() }
        addTagButton.setOnClickListener {
            launch {
                enableBackgroundTaskRunning()

                with(TagChooserDialog(this@TaggerActivity)) {
                    title = "Choose appropriate tag"
                    setCanceledOnTouchOutside(false)
                    show()
                }

                disableBackgroundTaskRunning()
            }
        }
        confirmButton.setOnClickListener {
            launch {
                enableBackgroundTaskRunning()

                updateStatistics(
                    predictionsChipGroup.children
                        .map { chip -> (chip as Chip).let { it.id to it.text.toString() } }
                        .toMap()
                )

                hideKeyboard(this@TaggerActivity.currentFocus ?: View(this@TaggerActivity))
                finish()
            }
        }
    }

    override fun onDestroy() {
        job.cancel()
        super.onDestroy()
    }

    override fun onResume() {
        super.onResume()

        imageUri = intent.extras?.get("ImageURI") as Uri?
        imageUri?.let { imageUri ->
            launch {
                enableBackgroundTaskRunning()

                val (hash, bitmap) = withContext(Dispatchers.Default) {
                    loadBitmap(imageUri, Constants.INPUT_WIDTH, Constants.INPUT_HEIGHT).let { bitmap ->
                        bitmap?.let { Utils.hashMd5(it) } to bitmap
                    }
                }

                hash?.let { md5 ->
                    val predictedClassIds = if (predictions.contains(md5)) {
                        predictions[md5]
                    } else {
                        bitmap?.let { bm ->
                            imageView.setImageBitmap(bm)

                            ImageTagger.cnn?.let { cnn ->
                                val embeddingVector = cnn.forward(bm)

                                val statistics = withContext(Dispatchers.IO) { statsTable.getStatistics() }
                                val x = Nd4j.create(embeddingVector).also { imageView.tag = it }
                                val classProbabilities = gnb.calculateProbabilities(x, statistics)

                                Utils.getTopPredictions(classProbabilities).also { topPredictions ->
                                    predictions[md5] = topPredictions
                                }
                            }
                        }
                    }

                    val predictedLabels = predictedClassIds?.let { predictedIds ->
                        statsTable.getStatisticsById(predictedIds)
                    }
                        ?.map { it.cid to it.label }
                        ?.toMap(mutableMapOf()) ?: mutableMapOf()

                    for ((classId, label) in predictedLabels) {
                        addNewChip(classId, label)
                    }
                }

                disableBackgroundTaskRunning()
            }
        }
    }

    override fun onBackPressed() {
        performCancelAction()
    }

    private suspend fun updateStatistics(associatedClasses: Map<Int, String>) = withContext(Dispatchers.IO) {
        val statisticsToPut = mutableListOf<Statistics>()

        val x = imageView.tag as INDArray
        val (existingIds, newIds) = associatedClasses.keys.partition { classId ->
            classId in statsTable.getClassIds()
        }

        if (existingIds.isNotEmpty()) {
            statsTable.getStatisticsById(existingIds)
                .map { classStatistics ->
                    val diff = x.sub(classStatistics.mean)
                    val inc = diff.mul(Constants.MOMENTUM)

                    classStatistics.mean?.assign(
                        classStatistics.mean.add(inc)
                    )
                    classStatistics.std?.assign(
                        classStatistics.std.add(diff.mul(inc)).mul(1.0f - Constants.MOMENTUM)
                    )
                    classStatistics.count = classStatistics.count?.inc()

                    classStatistics
                }
                .also { updatedStatistics -> statisticsToPut.addAll(updatedStatistics) }
        }

        if (newIds.isNotEmpty()) {
            newIds
                .map { classId -> classId to associatedClasses[classId] }
                .map { (classId, label) ->
                    Statistics(classId, label, x, Nd4j.ones(*x.shape()).mul(x.stdNumber()), 1)
                }
                .also { newStatistics -> statisticsToPut.addAll(newStatistics) }
        }

        if (statisticsToPut.isNotEmpty()) {
            statsTable.insertStatistics(statisticsToPut)
        }
    }

    private fun performCancelAction() {
        if (inferenceIsRunning) {
            interruptBackgroundTaskAlert()
        } else {
            exitIgnoreChangesAlert()
        }
    }

    private fun interruptBackgroundTaskAlert() {
        with(AlertDialog.Builder(this)) {
            setTitle("Interrupt running process?")
            setPositiveButton(android.R.string.ok) { _, _ ->
                job.cancel()
                this@TaggerActivity.disableBackgroundTaskRunning()
            }
            setNegativeButton(android.R.string.cancel) { _, _ -> }
            show()
        }
    }

    private fun exitIgnoreChangesAlert() {
        if (getConfirmButtonEnabled()) {
            with(AlertDialog.Builder(this)) {
                setTitle("Discard changes?")
                setMessage("Tags associated with the image will be lost")
                setPositiveButton(android.R.string.ok) { _, _ -> super.onBackPressed() }
                setNegativeButton(android.R.string.cancel) { _, _ -> }
                show()
            }
        } else {
            finish()
        }
    }

    fun addNewChip(classId: Int?, label: String?) {
        classId?.let {
            val classIdsAssigned = IntRange(0, predictionsChipGroup.childCount - 1)
                .map { idx -> predictionsChipGroup[idx].id }

            if (!classIdsAssigned.contains(classId)) {
                with(Chip(predictionsChipGroup.context)) {
                    id = classId
                    text = label
                    isCloseIconVisible = true
                    setOnCloseIconClickListener {
                        predictionsChipGroup.removeView(this)
                        confirmButton.isEnabled = getConfirmButtonEnabled()
                    }
                    predictionsChipGroup.addView(this)
                }
            }

            confirmButton.isEnabled = true
        }
    }

    private fun enableBackgroundTaskRunning() {
        inferenceIsRunning = true
        progressBar.visibility = View.VISIBLE
        addTagButton.isEnabled = false
        confirmButton.isEnabled = false
        predictionsChipGroup.forEach { chip ->
            (chip as Chip).isEnabled = false
        }
    }

    private fun disableBackgroundTaskRunning() {
        progressBar.visibility = View.INVISIBLE
        addTagButton.isEnabled = imageView.tag != null
        confirmButton.isEnabled = getConfirmButtonEnabled()
        predictionsChipGroup.forEach { chip ->
            (chip as Chip).isEnabled = true
        }
        inferenceIsRunning = false
    }

    private fun getConfirmButtonEnabled(): Boolean {
        return predictionsChipGroup.size > 0
    }

}
