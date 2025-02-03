# semantic-tagging-tools
simple tag expander & merger based on tags within your dataset

the model you use for synonym matching will HEAVILY influence the results you get.

## Overview
1. Group all unique tags, use a set() to make it easier
2. Pass the tags, in the format "x","y","z" to the embedding model (i used mistral-embed)
3. Normalise and pass the new embeddings to faiss
4. Adjust the threshold for matching within faiss
   4b. For each tag in your original list, the model will be supplied with the tag and candidate vectors from faiss which are within your similarity threshold (distance)
   4c. The model (i chose mistral-small v3) will return a json with valid synonyms it picked.
5. Rewrite the original label.
6. Profit?

## Thoughts
I used mistral-small as its free and good enough for the task, though it will make mistakes sometimes. If you used something bigger (like Qwen 32+B, or Claude) you will get better results, as they're smarter models.

i believe normalising before feeding the vectors to fass is *probably* uncesesary, but testing with it on seemed better than without

Currently, the program will go through the list and if necessary try to batch the candidate (for my testing i set batch size and k neighbours to 10 so it'll never show, please change this). this means that if word X has 200 candidates, and you have a batch size of 100, the llm would get two batches of 100 and add the synonyms at the end, though you might wanna mess with the semantic threshold a little so you dont get too many overlapping candidates.


info on faiss: https://github.com/facebookresearch/faiss/wiki/


Examples:

<img width="472" alt="ex1" src="https://github.com/user-attachments/assets/5bf28e02-3fba-4015-adae-77679a701643" />
<img width="651" alt="ex2" src="https://github.com/user-attachments/assets/d3af2f87-70e8-4ba5-a7d6-aee80c4a7285" />
<img width="657" alt="ex3" src="https://github.com/user-attachments/assets/a65b7fce-292f-4510-8a51-bd91e5054f87" />
