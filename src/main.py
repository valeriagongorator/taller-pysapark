import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 1. Crear sesión de Spark
spark = SparkSession.builder \
    .appName("Taller ETL Online Retail") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

DATA_PATH = "data/Online_Retail.csv"
OUTPUT_DIR = "src/outputs"

# Lectura y limpieza inicial
df = spark.read.option("header", "true").option("inferSchema", "true").csv(DATA_PATH)

df_cleaned = df.withColumn("Quantity", F.col("Quantity").cast("integer")) \
               .withColumn("UnitPrice", F.col("UnitPrice").cast("double")) \
               .withColumn("CustomerID", F.col("CustomerID").cast("integer")) \
               .withColumn("InvoiceDate", F.to_timestamp(F.col("InvoiceDate"))) \
               .withColumn("TotalSpent", F.col("Quantity") * F.col("UnitPrice"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Total facturas
df_q1 = spark.createDataFrame([(df_cleaned.select("InvoiceNo").distinct().count(),)], ["total_facturas"])
df_q1.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/q1_total_facturas.csv")

# 2. Clientes únicos
df_q2 = spark.createDataFrame([(df_cleaned.filter(F.col("CustomerID").isNotNull()).select("CustomerID").distinct().count(),)], ["total_clientes"])
df_q2.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/q2_clientes_unicos.csv")

# 3. Ingreso total
df_q3 = df_cleaned.agg(F.sum("TotalSpent").alias("ingreso_total"))
df_q3.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/q3_ingreso_total.csv")

# 4. Producto más vendido
df_q4 = df_cleaned.groupBy("StockCode", "Description").agg(F.sum("Quantity").alias("total_cantidad")).orderBy(F.desc("total_cantidad"))
df_q4.limit(10).coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/q4_producto_mas_vendido.csv")

# 5. Cliente con mayor volumen de compra
df_q5 = df_cleaned.filter(F.col("CustomerID").isNotNull()).groupBy("CustomerID").agg(F.sum("TotalSpent").alias("total_comprado")).orderBy(F.desc("total_comprado"))
df_q5.limit(10).coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/q5_top_clientes.csv")

# 6. Top 5 países fuera de Reino Unido
df_q6 = df_cleaned.filter(F.col("Country") != "United Kingdom").groupBy("Country").agg(F.sum("TotalSpent").alias("total_comprado")).orderBy(F.desc("total_comprado")).limit(5)
df_q6.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/q6_top_paises_fuera_uk.csv")

# 7. Ticket promedio por factura
df_facturas = df_cleaned.groupBy("InvoiceNo").agg(F.sum("TotalSpent").alias("total_factura"))
df_q7 = df_facturas.agg(F.avg("total_factura").alias("ticket_promedio"))
df_q7.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/q7_ticket_promedio.csv")

# 8. Mínimo, máximo y promedio por factura
df_q8 = df_cleaned.groupBy("InvoiceNo").agg(F.sum("Quantity").alias("total_productos")).agg(
    F.min("total_productos").alias("min_productos"),
    F.max("total_productos").alias("max_productos"),
    F.avg("total_productos").alias("promedio_productos")
)
df_q8.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/q8_metricas_facturas.csv")

# 9. Mes con más ventas
df_q9 = df_cleaned.withColumn("YearMonth", F.date_format("InvoiceDate", "yyyy-MM")).groupBy("YearMonth").agg(F.sum("TotalSpent").alias("total_ventas")).orderBy(F.desc("total_ventas"))
df_q9.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/q9_ventas_por_mes.csv")

# 10. Porcentaje de facturas con devoluciones
total_f = df_cleaned.select("InvoiceNo").distinct().count()
dev_f = df_cleaned.filter(F.col("Quantity") < 0).select("InvoiceNo").distinct().count()
df_q10 = spark.createDataFrame([((dev_f / total_f) * 100,)], ["porcentaje_devoluciones"])
df_q10.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{OUTPUT_DIR}/q10_porcentaje_devoluciones.csv")

print("Se han exportado exitosamente los 10 archivos CSV a src/outputs/")
spark.stop()