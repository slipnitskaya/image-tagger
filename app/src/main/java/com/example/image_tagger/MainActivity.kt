package com.example.image_tagger

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.FileProvider
import androidx.core.view.isVisible
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.floatingactionbutton.FloatingActionButton
import java.io.File
import java.io.FileOutputStream
import java.util.*

class MainActivity : AppCompatActivity() {

    private val permissionsRequired = listOf(Manifest.permission.CAMERA to Utils.Action.CAPTURE_IMAGE())
    private val imageExtensions = listOf("jpg", "png", "gif")
    private lateinit var imageFeed: RecyclerView
    private lateinit var uploadImageText: TextView
    private lateinit var fabUpload: FloatingActionButton
    private lateinit var fabCamera: FloatingActionButton
    lateinit var listOfImages: MutableList<File>

    private var imageCounter: Int
        get() = getPreferences(Context.MODE_PRIVATE).getInt(Constants.IMAGE_COUNTER_KEY, 0)
        set(value) {
            with(getPreferences(Context.MODE_PRIVATE).edit()) {
                putInt(Constants.IMAGE_COUNTER_KEY, value)
                apply()
            }
        }
    private lateinit var imageDir: File
    private lateinit var imageFile: File

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageFeed = findViewById(R.id.image_feed)
        uploadImageText = findViewById(R.id.upload_image_text)
        fabUpload = findViewById(R.id.fab_upload)
        fabCamera = findViewById(R.id.fab_camera)

        imageDir = File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), Constants.IMAGE_FOLDER_NAME)
        if (!imageDir.exists()) {
            imageDir.mkdirs()
        }

        fabUpload.setOnClickListener {
            val getIntent = Intent(Intent.ACTION_GET_CONTENT)
            getIntent.type = "image/*"

            val pickIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            pickIntent.type = "image/*"

            val chooserIntent = Intent.createChooser(getIntent, "Select Image")
            chooserIntent.putExtra(Intent.EXTRA_INITIAL_INTENTS, arrayOf(pickIntent))

            startActivityForResult(chooserIntent, Utils.Action.PICK_IMAGE())
        }

        fabCamera.setOnClickListener {
            val captureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)

            imageFile = getNewImageFile()
            val fileProvider = FileProvider.getUriForFile(
                    this, BuildConfig.APPLICATION_ID + ".provider", imageFile
            )
            captureIntent.putExtra(MediaStore.EXTRA_OUTPUT, fileProvider)

            startActivityForResult(captureIntent, Utils.Action.CAPTURE_IMAGE())
        }

        listOfImages = getImageFiles(imageDir).toMutableList()
        with(LinearLayoutManager(this)) {
            reverseLayout = true
            stackFromEnd = true
            imageFeed.layoutManager = this
        }
        imageFeed.adapter = ImageListAdapter(this)

        uploadImageText.isVisible = listOfImages.isEmpty()

        checkPermissions()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (Utils.Action.values().any { it() == requestCode } && resultCode == RESULT_OK) {
            when (requestCode) {
                Utils.Action.PICK_IMAGE() -> {
                    val imageUri = data?.data
                    imageUri?.let {
                        imageFile = getNewImageFile()
                        val bitmap = loadBitmap(it, Constants.INPUT_WIDTH, Constants.INPUT_HEIGHT)
                        FileOutputStream(imageFile).use { fos ->
                            bitmap?.compress(Bitmap.CompressFormat.PNG, 90, fos)
                        }
                    }
                }
                Utils.Action.CAPTURE_IMAGE() -> { }
            }
            listOfImages.clear()
            listOfImages.addAll(getImageFiles(imageDir).sorted())
            uploadImageText.isVisible = listOfImages.isEmpty()
            imageFeed.adapter?.notifyDataSetChanged()
            imageFeed.layoutManager?.scrollToPosition(listOfImages.size - 1)
        }
    }

    private fun checkPermissions() = permissionsRequired.forEach { (perm, action) ->
        if (checkSelfPermission(perm) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(perm), action)
        }
    }

    private fun getImageFiles(root: File): List<File> {
        val list = mutableListOf<File>()

        root.listFiles()?.forEach { file ->
            list.addAll(
                    when {
                        file.isDirectory -> {
                            getImageFiles(file)
                        }
                        file.extension.toLowerCase(Locale.getDefault()) in imageExtensions -> {
                            listOf(file)
                        }
                        else -> {
                            listOf()
                        }
                    }
            )
        }

        return list
    }

    private fun getNewImageFile(): File {
        val imageFilename = Constants.IMAGE_FILENAME + (imageCounter++).toString().padStart(5, '0') + ".jpg"
        return File(imageDir, imageFilename)
    }

}

class ImageListAdapter(private val parent: MainActivity) : RecyclerView.Adapter<ImageListAdapter.ImageListViewHolder>() {

    inner class ImageListViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val imageView: ImageView = itemView.findViewById(R.id.image_entry)

        init {
            imageView.setOnClickListener {
                val imageUri = it.tag as Uri
                val captionIntent = Intent(parent.baseContext, TaggerActivity::class.java)
                captionIntent.putExtra("ImageURI", imageUri)
                parent.startActivity(captionIntent)
            }
        }

        fun bind(imageFile: File) {
            val imageUri = Uri.fromFile(imageFile)
            val bitmap = parent.loadBitmap(imageUri, Constants.INPUT_WIDTH, Constants.INPUT_HEIGHT)
            imageView.setImageBitmap(bitmap)
            imageView.tag = imageUri
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ImageListViewHolder {
        return ImageListViewHolder(
                LayoutInflater.from(parent.context).inflate(R.layout.image_list_row, parent, false)
        )
    }

    override fun onBindViewHolder(holder: ImageListViewHolder, position: Int) {
        holder.bind(parent.listOfImages[position])
    }

    override fun getItemCount(): Int {
        return parent.listOfImages.size
    }

}
