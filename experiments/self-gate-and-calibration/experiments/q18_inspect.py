"""inspect the downloaded FlyWire table (data only; run with python -I, path passed as argument)"""
import sys
import pyarrow.feather as feather
p = sys.argv[1]
t = feather.read_table(p)
print("rows", t.num_rows, "columns", t.column_names)
print(t.schema)
df = t.slice(0, 5).to_pandas()
print(df.to_string())
