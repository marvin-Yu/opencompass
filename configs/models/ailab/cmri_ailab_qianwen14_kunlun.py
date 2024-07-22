from opencompass.models import AilabCommon

models = [
    dict(abbr='qwen14B_kunlun',
         type=AilabCommon,
         path='AilabCommon',
         model_id='xft',
         max_out_len=512,
         max_seq_len=32768,
         batch_size=1,
         query_per_second=1,
         stream=False,
         kunlun=True,
         mode='rear',
         url='http://127.0.0.1:8000/v1/chat/completions')
         
]