export CUDA_VISIBLE_DEVICES=0

#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh 

# Define folders
folders=(
"Sines_se_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0"
"VDP_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0"
"ECL_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0"
"Traffic_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0"
)
# Define methods
methods=(
"DeepAR"
"ARIMA"
)

for method in "${methods[@]}"; do
    if [[ "$method" == "ARIMA" ]]; then
        script="experiments/exp_arima.py"
        conda activate gluontsR
    else
        script="experiments/exp_gluonts.py"
        conda activate gluonts
    fi

    for folder in "${folders[@]}"; do
        prediction_length=96
        past_length=96
        mamba_batch_size=16

        # Special cases based on folder name
        if [[ "$folder" == *"Traffic"* ]]; then
            mamba_batch_size=8
        fi

        if [[ "$folder" == *"Brownian"* ]]; then
            prediction_length=128
            past_length=128
        fi

        # Select the script based on method
        echo "$script"

        # Call python with modified arguments
        python "$script" \
            --folder "$folder" \
            --Method "$method" \
            --prediction_length "$prediction_length" \
            --past_length "$past_length" \
            --mamba_batch_size "$mamba_batch_size"
    done
done
