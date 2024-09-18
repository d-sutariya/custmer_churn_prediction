# Data Directory Overview

The `data/` directory is organized into three subfolders that store the data at different stages of the project pipeline. Here’s an overview of each:

## Directory Structure

### 1. `raw/`
This folder stores the **raw, unprocessed data** directly as obtained from the source. No modifications or cleaning have been applied to this data yet.

### 2. `interim/`
The `interim/` folder contains **data that has been partially processed**. This stage typically includes initial cleaning steps, but the data is not yet fully transformed or feature-engineered. It's a temporary space before final transformations are applied.

### 3. `processed/`
The `processed/` folder stores the **final, fully cleaned, and transformed data**. This data is ready for modeling and analysis, having undergone all the necessary preprocessing steps like feature engineering, imputation, and normalization.

---

These folders are crucial for maintaining a clear structure in the data pipeline, ensuring that raw data is preserved, intermediate steps are tracked, and the final dataset is easily accessible for training and prediction purposes.
