import hanzo_ml as ml

def test_public_import_and_preprocessing():
    values=(ml.Tensor([10.0,20.0,0.0,0.0])/100.0).values()
    assert abs(values[0]-.1)<1e-6
    assert abs(values[1]-.2)<1e-6
    assert len(values)==4

def test_public_tensor_matmul():
    matrix=ml.Tensor([[1.,2.],[3.,4.]])
    assert matrix.matmul(matrix.t()).values()==[[5.,11.],[11.,25.]]
