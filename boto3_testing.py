import boto3
import json

# Let's use Amazon S3
s3 = boto3.resource('s3')
#
client = boto3.client("s3")
#
# #Print out bucket names
# for bucket in s3.buckets.all():
#     print(bucket.name)

# Upload a new file
# data = open('/home/neural/Documents/Sabeeha/codes/weights.onnx', 'rb')
# s3.Bucket('sabeeha-bucket').put_object(Key='weights.onnx', Body=data)
#
# iterating through all objects in the bucket
#bucket = s3.Bucket("sabeeha-bucket")
# buckets_items = list(bucket.objects.all())
# print(buckets_items)
# for item in buckets_items:
#     client.download_file("sabeeha-bucket",'weights.onnx','new_weights.onnx')


bucket = s3.Bucket("sabeeha-bucket")
buckets_items = list(bucket.objects.all())
print(buckets_items)
# To delete object from s3 bucket knowing its bucket name and key
s3 = boto3.resource('s3')
s3.Object('sabeeha-bucket', 'weights.onnx').delete()
#
buckets_items = list(bucket.objects.all())
print(buckets_items)

# # Alternatively to delete an object,we can use the following approach using client
# client = boto3.client('s3')
# client.delete_object(Bucket='mybucketname', Key='myfile.whatever')

#
# #Accessing the direct object using bucket name and key value
# obj = s3.Object(bucket_name='new-bucket-07f2294b', key='Mayo.jpg')
# print(obj)

#creating a bucket using s3
#s3.create_bucket(Bucket='sabeeha-bucket')