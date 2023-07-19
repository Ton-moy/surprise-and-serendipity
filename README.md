# Topic-Level Bayesian Surprise and Serendipity for Recommender Systems
This repository contains the code and dataset for the paper titled "Topic-Level Bayesian Surprise and Serendipity for Recommender Systems." The paper focuses on enhancing recommendation algorithms by incorporating topic-level Bayesian surprise and serendipity measures.

## Machine Configuration:
The experiments were conducted on a distributed system utilizing a cluster of multiple computers, typically ranging from 10 to 30, which operated simultaneously. Each computer had the following configuration:
- Dual 24-Core Intel Xeon Gold 6248R CPU @ 3.00GHz (48 cores per node)
- 384GB RAM (8GB per core)
- 100GBit EDR Infiniband Interconnect

Running the code on a personal computer or a single machine may significantly increase the processing time, possibly taking 5-10 days to complete some codes. So it is recommended to run the code on a distributed system if you have the opportunity. Please feel free to reach out to us regarding this.

## Experiment Replication:
To replicate our experiments, follow the steps below:
- #### Get Raw Dataset:
  - From the following link, download the raw dataset named "goodread_reviews_dedup.json.gz" that we have used in our experiments: [Raw Dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/reviews)
- #### Create the 'Books' Set:
  - Option 1: Use the provided code "create_Books_set.py" from the "data_creation" folder to create the "Books" set, where each file represents a book with a maximum of 10,000 tokens.
  - Option 2: Download the pre-created "Books" set from the following link: [Books](https://drive.google.com/file/d/1ymtN75HkxWKiLuFjez5KCLAf3K1SEkoD/view?usp=sharing)
- #### Create the 'Users' Set:
  - Option 1: Use the provided code "create_Users_set.py" from the "data_creation" folder to create the "Users" set, where each file represents a user's reading history. Each user includes the following fields: "book_id", "user_id", "rating", "timestring", "timestamp".
  - Option 2: Download the pre-created "Users" set from the following link: [Users](https://drive.google.com/file/d/1SY6zSqbxEdtrUmgk42Cg6p9OEz-mFMrU/view?usp=sharing)
- #### Train the LDA Model:
  - Option 1: After creating the dataset, train an LDA model using the books that contain at least 1000 tokens. You can use the provided code "train_lda_model.py"  
  - Option 2: Use our pre-trained model available in the following link: [pre_trained_lda_model](https://drive.google.com/file/d/1DBztAei7S2Pd3p4902u30PMApRA4ngiT/view?usp=sharing)
- #### Topic Distribution Creation:
  - Use the pre-trained LDA model to generate topic distributions for all the books in the "Books" set. Save the topic distributions in the "topic_distrs" folder. You can accomplish this by running the "create_topic_distr.py" file.
- #### Create User Preferences:
  - We generated user preferences for all 26,374 users using five different models separately, three of which are based on Bayesian Surprise. You can generate user preferences for each model using the provided code in the "create_user_preferences" folder.
- #### Surprising and Serendipitous Item Recommendation:
  - Run the codes from the "surprise" and "serendipity" folders to get the final results.

## Annotated Dataset:
The repository also includes the annotated dataset for four reference users. The files "User1.txt," "User2.txt," "User3.txt," and "User4.txt" contain human annotations for these users. Each file provides the following information in chronological order of the users' reading history:
- Book_number
- Book_id
- Book_title
- Human_identified_main_topic
- Summary
- Human decision
- Reason  

For books that are surprising in terms of their topic, the annotation for "Human decision" is "Surprising", otherwise, the annotation is "Not surprising."


##### If you have any questions or need assistance, please feel free to reach out to us.
