package com.example.image_tagger

import androidx.room.*
import com.daveanthonythomas.moshipack.MoshiPack
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

@Entity
data class Statistics(
    @PrimaryKey val cid: Int,
    @ColumnInfo(name = "label") val label: String?,
    @ColumnInfo(name = "mean") val mean: INDArray?,
    @ColumnInfo(name = "std") val std: INDArray?,
    @ColumnInfo(name = "counts") var count: Int?
)

@Dao
interface StatisticsDao {
    @Query("SELECT `cid` FROM `statistics`")
    fun getClassIds(): List<Int>

    @Query("SELECT COUNT(*) FROM `statistics`")
    fun getNumRows(): Int

    @Query("SELECT * FROM `statistics`")
    fun getStatistics(): List<Statistics>

    @Query("SELECT * FROM `statistics` WHERE `cid` IN (:classIds)")
    fun getStatisticsById(classIds: List<Int>): List<Statistics>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    @Transaction
    fun insertStatistics(classStatistics: List<Statistics>)
}

class Converters {
    companion object {
        val packer = MoshiPack()
    }

    @TypeConverter
    fun nDArrayFromBlob(value: ByteArray?): INDArray? {
        return value?.let {
            Nd4j.create(packer.unpack<FloatArray>(it))
        }
    }

    @TypeConverter
    fun nDArrayToBlob(value: INDArray?): ByteArray? {
        return value?.let {
            packer.packToByteArray(it.toFloatVector())
        }
    }
}

@Database(entities = [Statistics::class], version = 2)
@TypeConverters(Converters::class)
abstract class TagsDB : RoomDatabase() {
    abstract fun statisticsDao(): StatisticsDao
}
