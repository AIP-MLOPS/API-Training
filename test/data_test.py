from data.sdk.download_sdk import s3_download

s3_download(
    clearml_access_key='65592A380E9EB6F013881A57E0FE6389',
    clearml_secret_key='1FE51DAD67FB066710CF935911A84058FE9279B1122E0CC4719C505B932DAE81',
    clearml_host="http://144.172.105.98:30003",
    s3_access_key='3386LN5KA2OFQXPTYM9S',
    s3_secret_key='AALvi6KexAeSNCsOMRqDHTRf10BQzNyy5BQnGIfO',
    s3_endpoint_url='http://144.172.105.98:7000',
    dataset_name='test_training',
    absolute_path='./data',
    user_name='mlopsadminv2'
)