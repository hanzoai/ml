import hanzo

print(f"mkl:         {hanzo.utils.has_mkl()}")
print(f"accelerate:  {hanzo.utils.has_accelerate()}")
print(f"num-threads: {hanzo.utils.get_num_threads()}")
print(f"cuda:        {hanzo.utils.cuda_is_available()}")

t = hanzo.Tensor(42.0)
print(t)
print(t.shape, t.rank, t.device)
print(t + t)

t = hanzo.Tensor([3.0, 1, 4, 1, 5, 9, 2, 6])
print(t)
print(t + t)

t = t.reshape([2, 4])
print(t.matmul(t.t()))

print(t.to_dtype(hanzo.u8))
print(t.to_dtype("u8"))

t = hanzo.randn((5, 3))
print(t)
print(t.dtype)

t = hanzo.randn((16, 256))
quant_t = t.quantize("q6k")
dequant_t = quant_t.dequantize()
diff2 = (t - dequant_t).sqr()
print(diff2.mean_all())
