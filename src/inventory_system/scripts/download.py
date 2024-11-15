import gdown

model_url = "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"  
weights = "./osnet_x0_25_msmt17.pt" 

gdown.download(model_url, str(weights), quiet=False)