# Genetic Engineering Attribution Challenge
This repository contains the code for the [Genetic Engineering Attribution Challenge](https://www.drivendata.org/competitions/63/genetic-engineering-attribution/). The objective of this challenge is to predict the laboratory of origin for plasmid DNA sequences.

The competition saw 1211 competitors and our proposed CNN model was **ranked 14** in the private leaderboard for the competition with a **score of 0.9128** on the testset.

**Note:** Please look at the [competition website](https://www.drivendata.org/competitions/63/genetic-engineering-attribution/) for the data format
## Executing the program
1. Download data for the [competition](https://www.drivendata.org/competitions/63/genetic-engineering-attribution/page/165/)

2. Configure the data directory and other desired parametes in the utils/config.py file

3. Create n folds of the training data for K-fold Cross-Validation
    ```
    python utils/create_fold.py
    ```

4. Train the model
    ```
    python engine.py
    ```

5. Evaluate the model on validation set for n-folds
    ```
    python pred_val.py
    ```

6. Make predictions for test_set
    ```
    python pred_test.py
    ```

7. Create submission file
    ```
    python create_submission.py
    ```
    Note: The submission script finds the top10 labels and assigns equal probability among them to reduce the final file size.

## Contact

For more details, please contact:

 - Seyedmostafa Sheikhalishahi: ssheikhalishahi@fbk.eu <br>
 - Vevake Balaraman: balaraman@fbk.eu