export CUDA_VISIBLE_DEVICES=0

#!/bin/bash

# Define folders
folders=(
"ECL_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0"
"Sines_se_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0"
"Traffic_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0"
"VDP_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0"
)


folders=(
"Traffic_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0"
"VDP_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0"
)
# Define methods
methods=(
"DeepAR"
)

#Run over all folder-method combinations
for method in "${methods[@]}"; do
    for folder in "${folders[@]}"; do
        python experiments/exp_gluonts.py --folder "$folder" --Method "$method"
    done
done