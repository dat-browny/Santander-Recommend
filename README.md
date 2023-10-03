# Santander-Recommend


## Instal dependencies
```
pip install numpy scikit-learn pandas kaggle
pip install git+https://github.com/gbolmier/funk-svd
pip install scikit-surprise
```
## Để tải dữ liệu, thực hiện theo các bước sau đây:
Try cập https://www.kaggle.com/settings và đăng nhập, ở mục API ấn vào nút Create New Token, sau đó hệ thống sẽ tải về file kaggle.json
Lấy file path của file kaggle.json sau đó thực hiện lệnh sau trong terminal
```
mkdir ~/.kaggle
mv #PATH/TO/KAGGLE.json ~/.kaggle/
```
Download script:
```
kaggle competitions download -c santander-product-recommendation
```
Unzip các file
```
unzip santander-production-recommendation.zip
unzip test_ver2.csv.zip
unzip train_ver2.csv.zip
```

## Submit CLI
Mỗi lần `get_submit_file` có thể submit trực tiếp lên kaggle bằng command line sau
```
kaggle competitions submit -c santander-product-recommendation -f submission.csv -m "Message"
```
Truy cập vào `https://www.kaggle.com/competitions/santander-product-recommendation/submissions` để xem kết quả.