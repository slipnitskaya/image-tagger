package com.example.image_tagger

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Base64
import android.view.View
import android.view.inputmethod.InputMethodManager
import androidx.room.Room
import com.daveanthonythomas.moshipack.MoshiPack
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import java.io.FileOutputStream

class ImageTagger : Application() {

    companion object {
        lateinit var mediaFolder: File
        lateinit var db: TagsDB
        var cnn: CNN? = null
    }

    override fun onCreate() {
        super.onCreate()

        db = Room
                .databaseBuilder(
                        applicationContext,
                        TagsDB::class.java,
                        Constants.DATABASE_NAME
                )
                .allowMainThreadQueries()
                .enableMultiInstanceInvalidation()
                .build()
        mayBePopulateDb()
        cnn = loadCNN()

        mediaFolder = File(externalCacheDir ?: cacheDir, Constants.IMAGE_FOLDER_NAME)
        if (mediaFolder.exists() && !mediaFolder.isDirectory) {
            mediaFolder.delete()
        }
        if (!mediaFolder.exists()) {
            mediaFolder.mkdirs()
        }
    }

    private fun mayBePopulateDb() {
        val statisticsTable = db.statisticsDao()
        if (statisticsTable.getNumRows() == 0) {
            val lines = assets.open(Constants.CSV_NAME)
                    .use {
                        it.readBytes().toString(Charsets.UTF_8)
                    }
                    .replace("\r", "")
                    .split("\n")

            val msgPack = MoshiPack()
            val classStatistics = mutableListOf<Statistics>()
            lines.filter {
                line -> line.isNotEmpty()
            }.forEach { line ->
                val cells = line.split(";")

                val classId = cells[0].toInt()
                val label = cells[1]
                val mean = Nd4j.create(msgPack.unpack<FloatArray>(Base64.decode(cells[2], Base64.DEFAULT)))
                val std = Nd4j.create(msgPack.unpack<FloatArray>(Base64.decode(cells[3], Base64.DEFAULT)))
                val count = cells[4].toInt()

                val statistics = Statistics(classId, label, mean, std, count)
                classStatistics.add(statistics)
            }

            statisticsTable.insertStatistics(classStatistics)
        }
    }

    private fun loadCNN(): CNN? {
        val tempDir = externalCacheDir ?: cacheDir
        val tempFile = File(tempDir, Constants.MODEL_NAME)
        if (!tempFile.exists() && tempFile.createNewFile()) {
            FileOutputStream(tempFile).use { fos ->
                assets.open(Constants.MODEL_NAME).use { fis ->
                    fis.copyTo(fos)
                }
            }
        }

        return if (tempFile.exists()) CNN(tempFile.absolutePath) else null
    }

}

fun Context.loadBitmap(imageUri: Uri, reqWidth: Int, reqHeight: Int): Bitmap? {
    return BitmapFactory.Options().run {
        inJustDecodeBounds = true
        contentResolver.openInputStream(imageUri).use {
            BitmapFactory.decodeStream(it, null, this)
        }

        inSampleSize = 1
        if (outHeight > reqHeight || outWidth > reqWidth) {
            val halfHeight: Int = outHeight / 2
            val halfWidth: Int = outWidth / 2

            // Calculate the largest inSampleSize value that is a power of 2 and keeps both
            // height and width larger than the requested height and width.
            while (halfHeight / inSampleSize >= reqHeight
                    && halfWidth / inSampleSize >= reqWidth) {
                inSampleSize *= 2
            }
        }

        inJustDecodeBounds = false
        contentResolver.openInputStream(imageUri).use {
            BitmapFactory.decodeStream(it, null, this)
        }
    }
}

fun Context.showKeyboard() {
    with(getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager) {
        toggleSoftInput(InputMethodManager.SHOW_FORCED, InputMethodManager.HIDE_IMPLICIT_ONLY)
    }
}

fun Context.hideKeyboard(view: View) {
    with(getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager) {
        hideSoftInputFromWindow(view.windowToken, 0)
    }
}
