#%%
# Extracting the data from the varioues source provided
import time 
import requests

import asyncio
import httpx


urls = [
    "https://books.toscrape.com/catalogue/category/books/travel_2/index.html",
    "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
    "https://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html",
    "https://books.toscrape.com/catalogue/category/books/sequential-art_5/index.html",
    ]



def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        output = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Execution end at {end_time - start_time}(s)")
        return output
    return wrapper

@timer
def fetch(urls):
    results = [requests.get(url) for url in urls]
    print(results)

@timer
async def fetch_parallel(urls):
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)

        # Process the responses
        for response in responses:
            print(f"URL: {response.url}")
            print(f"Status: {response.status_code}")
            print(f"Content (truncated): {response.text[:100]}")
            print("-" * 60)



if __name__ == "__main__":
    # fetch(urls)
    # asyncio.run(fetch_parallel(urls))
    # await fetch_parallel(urls) # in jupytor notebook if to show