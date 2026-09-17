import os
import shutil
import glob
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("Taller ETL Online Retail") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

DATA_PATH = "data/Online_Retail.csv"
OUTPUT_DIR = "src/outputs"
TEMP_DIR = "src/outputs/temp"

# Lectura explícita usando spark.read.format("csv")
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(DATA_PATH)

# Limpieza inicial y columnas derivadas
df_cleaned = df.withColumn("Quantity", F.col("Quantity").cast("integer")) \
               .withColumn("UnitPrice", F.col("UnitPrice").cast("double")) \
               .withColumn("CustomerID", F.col("CustomerID").cast("integer")) \
               .withColumn("InvoiceDate", F.to_timestamp(F.col("InvoiceDate"))) \
               .withColumn("TotalSpent", F.col("Quantity") * F.col("UnitPrice"))

os.makedirs(TEMP_DIR, exist_ok=True)

# Definición de análisis
# 1. Total facturas
df_q1 = spark.createDataFrame([(df_cleaned.select("InvoiceNo").distinct().count(),)], ["total_facturas"])

# 2. Clientes únicos
df_q2 = spark.createDataFrame([(df_cleaned.filter(F.col("CustomerID").isNotNull()).select("CustomerID").distinct().count(),)], ["total_clientes"])

# 3. Ingreso total
df_q3 = df_cleaned.agg(F.sum("TotalSpent").alias("ingreso_total"))

# 4. Producto más vendido en cantidad + JOIN
df_catalogo = df_cleaned.select("StockCode", "Description").distinct()
df_cantidades = df_cleaned.groupBy("StockCode").agg(F.sum("Quantity").alias("total_cantidad"))
df_q4 = df_cantidades.join(df_catalogo, on="StockCode", how="inner").select("StockCode", "Description", "total_cantidad").orderBy(F.desc("total_cantidad")).limit(10)

# 5. Cliente con mayor volumen de compra
df_q5 = df_cleaned.filter(F.col("CustomerID").isNotNull()).groupBy("CustomerID").agg(F.sum("TotalSpent").alias("total_comprado")).orderBy(F.desc("total_comprado")).limit(10)

# 6. Top 5 países fuera de Reino Unido
df_q6 = df_cleaned.where(F.col("Country") != "United Kingdom").groupBy("Country").agg(F.sum("TotalSpent").alias("total_comprado")).orderBy(F.desc("total_comprado")).limit(5)

# 7. Ticket promedio por factura
df_facturas = df_cleaned.groupBy("InvoiceNo").agg(F.sum("TotalSpent").alias("total_factura"))
df_q7 = df_facturas.agg(F.avg("total_factura").alias("ticket_promedio"))

# 8. Mínimo, máximo y promedio por factura
df_q8 = df_cleaned.groupBy("InvoiceNo").agg(F.sum("Quantity").alias("total_productos")).agg(
    F.min("total_productos").alias("min_productos"),
    F.max("total_productos").alias("max_productos"),
    F.avg("total_productos").alias("promedio_productos")
)

# 9. Mes con más ventas
df_q9 = df_cleaned.withColumn("YearMonth", F.date_format("InvoiceDate", "yyyy-MM")).groupBy("YearMonth").agg(F.sum("TotalSpent").alias("total_ventas")).orderBy(F.desc("total_ventas")).limit(1)

# 10. Porcentaje de facturas con devoluciones
total_f = df_cleaned.select("InvoiceNo").distinct().count()
dev_f = df_cleaned.filter(F.col("Quantity") < 0).select("InvoiceNo").distinct().count()
df_q10 = spark.createDataFrame([((dev_f / total_f) * 100,)], ["porcentaje_devoluciones"])

# Función para exportar sólo el archivo .csv limpio
def export_clean_csv(df, name):
    temp_folder = os.path.join(TEMP_DIR, name)
    final_file = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.coalesce(1).write.mode("overwrite").option("header", "true").format("csv").save(temp_folder)
    csv_file = glob.glob(f"{temp_folder}/part-*.csv")[0]
    shutil.move(csv_file, final_file)

exports = [
    (df_q1, "q1_total_facturas"),
    (df_q2, "q2_clientes_unicos"),
    (df_q3, "q3_ingreso_total"),
    (df_q4, "q4_producto_mas_vendido"),
    (df_q5, "q5_top_clientes"),
    (df_q6, "q6_top_paises_fuera_uk"),
    (df_q7, "q7_ticket_promedio"),
    (df_q8, "q8_metricas_facturas"),
    (df_q9, "q9_ventas_por_mes"),
    (df_q10, "q10_porcentaje_devoluciones")
]

for df_item, name in exports:
    export_clean_csv(df_item, name)

shutil.rmtree(TEMP_DIR)
spark.stop()
print("Archivos CSV exportados de forma limpia en src/outputs/")