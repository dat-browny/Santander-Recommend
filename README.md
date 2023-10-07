# Santander-Recommend


## Install dependencies
```
pip install -r requirements.txt
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
unzip santander-product-recommendation.zip
unzip test_ver2.csv.zip
unzip train_ver2.csv.zip
rm *.zip
```
## Transformers
Với mạng Transformers, trước khi huấn luyện phải chạy xử lý dữ liệu trước, tuy nhiên bước này khá mất thời gian, việc xử lý dữ liệu đã được làm trước và tải lên drive.

Xử lý dữ liệu từ đầu, bảo đảm trong directory chứa 2 file `train_ver2.csv` và `test_ver2.csv`, quá trình mất ~2h tuỳ thuộc vào phần cứng của từng máy.
```
python src/data.py
```
Tải dữ liệu đã xử lý và unpack (highly recommend):
```
gdown --fuzzy https://drive.google.com/file/d/1Ay7h5YDsegNt94Xf0DjfwDpPgr9o1xBt/view?usp=drive_link
unzip data.zip
rm data.zip
```
Sau khi đã có dữ liệu, training và inference mô hình transformers (sau khi chạy xong sẽ tự động có file `submission.csv`)
```
python src/bst.py
```

## Submit CLI
Mỗi lần `get_submit_file` có thể submit trực tiếp lên kaggle bằng command line sau
```
kaggle competitions submit -c santander-product-recommendation -f submission.csv -m "Message"
```
Truy cập vào `https://www.kaggle.com/competitions/santander-product-recommendation/submissions` để xem kết quả.
