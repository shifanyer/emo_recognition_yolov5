import deeplake
ds = deeplake.load('hub://activeloop/fer2013-train')
# print(ds)
ds.save_view(path="data2")