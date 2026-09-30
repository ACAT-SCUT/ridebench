1. Clone the **RideBench** repository and enter the repository root:

    ```bash
    git clone https://www.github.com/xxx/ridebench
    cd ridebench
    ```

2. Place the dataset into the default directory:

    ```bash
    mv /path/to/ride_hailing.csv ./datasets/ride_hailing.csv
    ```

3. Configure the environment:

    ```bash
    uv sync
    ```

    or

    ```bash
    pip install -r requirements.txt
    ```
    
4. Reproduce the experiments:

    * Regular Forecasting: `scripts/benchmark`
    * Long-Horizon Forecasting: `scripts/benchmark/ultra_long_forecast`
    * Area Scaling Study: `scripts/study_area`
    * Lookback Study: `scripts/study_lookback`
    * Loss Function Study: `scripts/study_loss`
    
    For example:
    
    ```bash
    bash ./scripts/benchmark/simple/SeasonalNaive.sh
    ```

5. One-click run:
    
    ```bash
    # Reproduce all the experiments above at once
    bash ./scripts/rochestrator_runner/all.sh
    ```