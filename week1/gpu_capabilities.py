import torch, time

print("torch      :", torch.__version__)
print("built cuda :", torch.version.cuda)          # 휠이 어떤 CUDA로 빌드됐나
print("available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    i = torch.cuda.current_device()
    print("device     :", torch.cuda.get_device_name(i))
    print("capability :", torch.cuda.get_device_capability(i))  # (8, 6) 이런 식
    print("arch list  :", torch.cuda.get_arch_list())  # 여기에 내 sm_XX 있어야 함

    # 실제 커널이 도는지까지 확인 (is_available()만으론 부족)
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    t = time.time(); (a @ b); torch.cuda.synchronize()
    print(f"gpu matmul : {time.time()-t:.4f}s")

    c = torch.randn(4096, 4096); d = torch.randn(4096, 4096)
    t = time.time(); (c @ d)
    print(f"cpu matmul : {time.time()-t:.4f}s")