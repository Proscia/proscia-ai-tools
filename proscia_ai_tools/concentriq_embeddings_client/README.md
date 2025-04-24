# Concentriq Embeddings Client

This is a Python client for the Concentriq Embeddings service. The client provides methods to submit jobs, estimate job duration, get job status, and fetch results.

## Usage

The client provides the following methods:

- `submit_job(data: Dict) -> SubmissionResponse`: Method to submit a job to the embeddings service.
- `estimate_job_cost(data: Dict) -> EstimationResponse`: Method to estimate a job cost.
- `get_job_status(ticket: str) -> StatusResponse`: Method to get the status of a job.
- `fetch_results(ticket: str, offset: int = 0, limit: int = 100) -> JobOutput`: Method to fetch results of a job.
- `poll_for_completion_and_fetch_results(ticket: str, check_interval: int = 5) -> JobOutput`: Polls job status and fetches results once complete.

Here is an example of how to use the client

### Initialize the client

```python
In [1]: from concentriq_embeddings_client import ConcentriqEmbeddingsClient

In [2]: client = ConcentriqEmbeddingsClient(
   ...:     base_url="https://some.concentriq.proscia.com",
   ...:     email="some.user@proscia.com",
   ...:     password="some-password"
   ...: )
```

### Create Payload

```python
In [3]: data = {
   ...:     "input_type": "image_ids",
   ...:     "input": [1,2,3],
   ...:     "mpp": 10.0,
   ...:     "model": "facebook/dinov2-base"
   ...: }
```

alternatively for repositories

```python
In [3]: data = {
   ...:     "input_type": "repository_ids",
   ...:     "input": [1918],
   ...:     "mpp": 10.0,
   ...:     "model": "facebook/dinov2-base"
   ...: }
```

### Estimate Job Cost

```python
In [4]: estimate_job_response = client.estimate_job_cost(data)

In [5]: estimate_job_response
Out[5]: EstimationResponse(job_duration_m=100.23184276163612, before_job_m=98582.71392620936, after_job_m=98482.48208344771, num_invalid_images=None, invalid_image_ids=None)
```

### Submit Job

```python
In [6]: submit_job_response = client.submit_job(data)

In [7]: submit_job_response
Out[7]: SubmissionResponse(ticket_id='cf1c238a-d915-49f7-aa19-32aa115e6cd6', job_duration_m=100.23184276163612, before_job_m=98582.71392620936, after_job_m=98482.48208344771)
```

### Monitor Job Progress

```python
In [8]: status_response = client.get_job_status(submit_job_response.ticket_id)

In [9]: status_response
Out[9]: StatusResponse(status='queued', progress=0.0, finished=0, failed=0, queued=252, processing=0)
```

### Get Results

```python
In [10]: results = client.fetch_results(submit_job_response.ticket_id)
In [11]: results
Out[11]: JobOutput(images=[ImageOutput(image_id=6564, repository_id=None, status='finished', model='facebook/dinov2-base', patch_size=224, grid_rows=12, grid_cols=11, pad_height=118, pad_width=221, mpp=10.0, embeddings_url='https://embeddings-api.s3.amazonaws.com/output/f27a5f50-a298-4761-9dd7-8eafc25bbb90_6564.safetensors?AWSAccessKeyId=AXXXXXXXX', thumb_url='https://embeddings-api.s3.amazonaws.com/output/f27a5f50-a298-4761-9dd7-8eafc25bbb90_6564.png?AWSAccessKeyId=AXXXXXXXX', thumb_mpp=7.0])
```
