import boto3

class S3AudioFile(object):
    def __init__(self, aws_profile: str, video_id: str, local_path: str, bucket_name: str):
        self.video_id = video_id
        self.local_path = local_path
        self.bucket_name = bucket_name
        self.s3_client = boto3.Session(profile_name=aws_profile).client('s3')
        
    def __enter__(self):
        print(f'Uploading {self.local_path} to S3 bucket {self.bucket_name} as {self.video_id}.wav')
        self.s3_client.upload_file(self.local_path, self.bucket_name, f'{self.video_id}.wav')
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        print(f'Deleting {self.video_id}.wav from S3 bucket {self.bucket_name}')
        self.s3_client.delete_object(Bucket=self.bucket_name, Key=f'{self.video_id}.wav')
        
    def public_url(self) -> str:
        return f'https://{self.bucket_name}.s3.amazonaws.com/{self.video_id}.wav'
