package com.example.caption_assistant

import android.database.sqlite.SQLiteDatabase
import android.net.Uri
import android.os.Bundle
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import com.daveanthonythomas.moshipack.MoshiPack
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import java.io.FileOutputStream


class CaptionActivity : AppCompatActivity() {

    private var imageUri: Uri? = null
    private lateinit var imageView: ImageView
    private lateinit var classMean: INDArray
    private lateinit var classStd: INDArray

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_caption)

        imageView = findViewById(R.id.chosen_image)
        val (classMean_, classStd_) = loadStatisticsFromDB()
        classMean = classMean_
        classStd = classStd_
    }

    override fun onStart() {
        super.onStart()

        imageUri = intent.extras?.get("ImageURI") as Uri?
        if (imageUri != null) {
            imageView.setImageURI(imageUri)
        }
    }

    private fun loadStatisticsFromDB(): Pair<INDArray, INDArray> {
        val msgPack = MoshiPack()

        val tempDir = externalCacheDir ?: cacheDir
        val tempFile = File(tempDir, "caption_assistant.db")
        if (!tempFile.exists()) {
            if (tempFile.createNewFile()) {
                FileOutputStream(tempFile).use { fos ->
                    assets.open("caption_assistant.db").use { fis ->
                        fis.copyTo(fos)
                    }
                }
            }
        }

        var classMean: INDArray?
        var classStd: INDArray?
        val classLabels = HashMap<Long, String>()
        SQLiteDatabase
            .openDatabase(tempFile, SQLiteDatabase.OpenParams.Builder().build())
            .use { db ->
                db.query(
                    "statistics",
                    arrayOf("cid", "label", "mean", "std"),
                    null,
                    null,
                    null,
                    null,
                    null
                ).use { cur ->
                    classMean = Nd4j.zeros(cur.count, 512).castTo(DataType.FLOAT)
                    classStd = Nd4j.zeros(cur.count, 512).castTo(DataType.FLOAT)
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

                            classMean!!.putRow(classId, mean)
                            classStd!!.putRow(classId, std)
                            classLabels[classId] = label
                        }
                }
            }

        return Pair(classMean!!, classStd!!)
    }
}
