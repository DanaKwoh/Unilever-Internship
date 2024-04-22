data {
	int<lower=0> N;
	vector[N] Dove_Bar_Offline_Volume;
	vector[N] Dove_Bar_Offline_Average_Price;
	vector[N] Safeguard_PW_TV_GRP;
	vector[N] Dove_Bar_Offline_TDP_Distribution;
	vector[N] Dove_PW_MSP_TV_GRP;
	vector[N] DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_OOH_ownM_0370;
	vector[N] DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_PREROLL_ownM_0370;
	vector[N] DOVE_Shampoo_ALL_ALL_ALL_CDS_DIGITAL_ACTIVATION_haloD_0330;
	vector[N] DOVE_Shampoo_ALL_ALL_ALL_CDS_SOCIAL_haloD_0330;
	vector[N] DOVE_ALL_BAR_OFFLINE_RETAIL_CDP_ownpromo_L2;
	vector[N] DOVE_ALL_BAR_OFFLINE_RETAIL_BBP_RT_ownpromo_L2;
	vector[N] DOVE_OTHERS_BAR_OFFLINE_RETAIL_TCP_DM_ownpromo_L2;
	vector[N] DOVE_OTHERS_BAR_OFFLINE_RETAIL_CPP_ownpromo_L2;
	vector[N] DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_OOH_haloM_0370;
	vector[N] DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_PREROLL_haloM_0370;
	vector[N] SET1_RETAIL;
	vector[N] SET1_ONLINE;
}

parameters {
	real<upper=0> b_Dove_Bar_Offline_Average_Price;
	real<upper=0> b_Safeguard_PW_TV_GRP;
	real<lower=0> b_Dove_Bar_Offline_TDP_Distribution;
	real<lower=0> b_Dove_PW_MSP_TV_GRP;
	real<lower=0> b_DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_OOH_ownM_0370;
	real<lower=0> b_DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_PREROLL_ownM_0370;
	real<lower=0> b_DOVE_Shampoo_ALL_ALL_ALL_CDS_DIGITAL_ACTIVATION_haloD_0330;
	real<lower=0> b_DOVE_Shampoo_ALL_ALL_ALL_CDS_SOCIAL_haloD_0330;
	real<lower=0> b_DOVE_ALL_BAR_OFFLINE_RETAIL_CDP_ownpromo_L2;
	real<lower=0> b_DOVE_ALL_BAR_OFFLINE_RETAIL_BBP_RT_ownpromo_L2;
	real<lower=0> b_DOVE_OTHERS_BAR_OFFLINE_RETAIL_TCP_DM_ownpromo_L2;
	real<lower=0> b_DOVE_OTHERS_BAR_OFFLINE_RETAIL_CPP_ownpromo_L2;
	real<lower=0> b_DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_OOH_haloM_0370;
	real<lower=0> b_DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_PREROLL_haloM_0370;
	real<lower=0> b_SET1_RETAIL;
	real<lower=0> b_SET1_ONLINE;

	real beta;
	real<lower=0> sigma;
}

model {
	b_Dove_Bar_Offline_Average_Price ~ normal(0, 1);
	b_Safeguard_PW_TV_GRP ~ normal(0, 1);
	b_Dove_Bar_Offline_TDP_Distribution ~ normal(0, 1);
	b_Dove_PW_MSP_TV_GRP ~ normal(0, 1);
	b_DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_OOH_ownM_0370 ~ normal(0, 1);
	b_DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_PREROLL_ownM_0370 ~ normal(0, 1);
	b_DOVE_Shampoo_ALL_ALL_ALL_CDS_DIGITAL_ACTIVATION_haloD_0330 ~ normal(0, 1);
	b_DOVE_Shampoo_ALL_ALL_ALL_CDS_SOCIAL_haloD_0330 ~ normal(0, 1);
	b_DOVE_ALL_BAR_OFFLINE_RETAIL_CDP_ownpromo_L2 ~ normal(0, 1);
	b_DOVE_ALL_BAR_OFFLINE_RETAIL_BBP_RT_ownpromo_L2 ~ normal(0, 1);
	b_DOVE_OTHERS_BAR_OFFLINE_RETAIL_TCP_DM_ownpromo_L2 ~ normal(0, 1);
	b_DOVE_OTHERS_BAR_OFFLINE_RETAIL_CPP_ownpromo_L2 ~ normal(0, 1);
	b_DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_OOH_haloM_0370 ~ normal(0, 1);
	b_DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_PREROLL_haloM_0370 ~ normal(0, 1);
	b_SET1_RETAIL ~ normal(0, 1);
	b_SET1_ONLINE ~ normal(0, 1);

	beta ~ normal(0, 1);
	sigma ~ cauchy(0, 5);

	Dove_Bar_Offline_Volume ~ normal(beta + b_Dove_Bar_Offline_Average_Price * Dove_Bar_Offline_Average_Price + b_Safeguard_PW_TV_GRP * Safeguard_PW_TV_GRP + b_Dove_Bar_Offline_TDP_Distribution * Dove_Bar_Offline_TDP_Distribution + b_Dove_PW_MSP_TV_GRP * Dove_PW_MSP_TV_GRP + b_DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_OOH_ownM_0370 * DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_OOH_ownM_0370 + b_DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_PREROLL_ownM_0370 * DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_PREROLL_ownM_0370 + b_DOVE_Shampoo_ALL_ALL_ALL_CDS_DIGITAL_ACTIVATION_haloD_0330 * DOVE_Shampoo_ALL_ALL_ALL_CDS_DIGITAL_ACTIVATION_haloD_0330 + b_DOVE_Shampoo_ALL_ALL_ALL_CDS_SOCIAL_haloD_0330 * DOVE_Shampoo_ALL_ALL_ALL_CDS_SOCIAL_haloD_0330 + b_DOVE_ALL_BAR_OFFLINE_RETAIL_CDP_ownpromo_L2 * DOVE_ALL_BAR_OFFLINE_RETAIL_CDP_ownpromo_L2 + b_DOVE_ALL_BAR_OFFLINE_RETAIL_BBP_RT_ownpromo_L2 * DOVE_ALL_BAR_OFFLINE_RETAIL_BBP_RT_ownpromo_L2 + b_DOVE_OTHERS_BAR_OFFLINE_RETAIL_TCP_DM_ownpromo_L2 * DOVE_OTHERS_BAR_OFFLINE_RETAIL_TCP_DM_ownpromo_L2 + b_DOVE_OTHERS_BAR_OFFLINE_RETAIL_CPP_ownpromo_L2 * DOVE_OTHERS_BAR_OFFLINE_RETAIL_CPP_ownpromo_L2 + b_DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_OOH_haloM_0370 * DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_OOH_haloM_0370 + b_DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_PREROLL_haloM_0370 * DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_PREROLL_haloM_0370 + b_SET1_RETAIL * SET1_RETAIL + b_SET1_ONLINE * SET1_ONLINE, sigma);
}

generated quantities {
	real prediction[N];
	real log_lik[N];
	for (n in 1:N)
		prediction[n] = normal_rng(beta + b_Dove_Bar_Offline_Average_Price * Dove_Bar_Offline_Average_Price[n] + b_Safeguard_PW_TV_GRP * Safeguard_PW_TV_GRP[n] + b_Dove_Bar_Offline_TDP_Distribution * Dove_Bar_Offline_TDP_Distribution[n] + b_Dove_PW_MSP_TV_GRP * Dove_PW_MSP_TV_GRP[n] + b_DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_OOH_ownM_0370 * DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_OOH_ownM_0370[n] + b_DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_PREROLL_ownM_0370 * DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_PREROLL_ownM_0370[n] + b_DOVE_Shampoo_ALL_ALL_ALL_CDS_DIGITAL_ACTIVATION_haloD_0330 * DOVE_Shampoo_ALL_ALL_ALL_CDS_DIGITAL_ACTIVATION_haloD_0330[n] + b_DOVE_Shampoo_ALL_ALL_ALL_CDS_SOCIAL_haloD_0330 * DOVE_Shampoo_ALL_ALL_ALL_CDS_SOCIAL_haloD_0330[n] + b_DOVE_ALL_BAR_OFFLINE_RETAIL_CDP_ownpromo_L2 * DOVE_ALL_BAR_OFFLINE_RETAIL_CDP_ownpromo_L2[n] + b_DOVE_ALL_BAR_OFFLINE_RETAIL_BBP_RT_ownpromo_L2 * DOVE_ALL_BAR_OFFLINE_RETAIL_BBP_RT_ownpromo_L2[n] + b_DOVE_OTHERS_BAR_OFFLINE_RETAIL_TCP_DM_ownpromo_L2 * DOVE_OTHERS_BAR_OFFLINE_RETAIL_TCP_DM_ownpromo_L2[n] + b_DOVE_OTHERS_BAR_OFFLINE_RETAIL_CPP_ownpromo_L2 * DOVE_OTHERS_BAR_OFFLINE_RETAIL_CPP_ownpromo_L2[n] + b_DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_OOH_haloM_0370 * DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_OOH_haloM_0370[n] + b_DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_PREROLL_haloM_0370 * DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_PREROLL_haloM_0370[n] + b_SET1_RETAIL * SET1_RETAIL[n] + b_SET1_ONLINE * SET1_ONLINE[n], sigma);

	for (n in 1:N)
		log_lik[n] = normal_lpdf(Dove_Bar_Offline_Volume[n] | beta + b_Dove_Bar_Offline_Average_Price * Dove_Bar_Offline_Average_Price[n] + b_Safeguard_PW_TV_GRP * Safeguard_PW_TV_GRP[n] + b_Dove_Bar_Offline_TDP_Distribution * Dove_Bar_Offline_TDP_Distribution[n] + b_Dove_PW_MSP_TV_GRP * Dove_PW_MSP_TV_GRP[n] + b_DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_OOH_ownM_0370 * DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_OOH_ownM_0370[n] + b_DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_PREROLL_ownM_0370 * DOVE_ALL_ALL_ALL_DIGITAL_SPENDS_MSP_PREROLL_ownM_0370[n] + b_DOVE_Shampoo_ALL_ALL_ALL_CDS_DIGITAL_ACTIVATION_haloD_0330 * DOVE_Shampoo_ALL_ALL_ALL_CDS_DIGITAL_ACTIVATION_haloD_0330[n] + b_DOVE_Shampoo_ALL_ALL_ALL_CDS_SOCIAL_haloD_0330 * DOVE_Shampoo_ALL_ALL_ALL_CDS_SOCIAL_haloD_0330[n] + b_DOVE_ALL_BAR_OFFLINE_RETAIL_CDP_ownpromo_L2 * DOVE_ALL_BAR_OFFLINE_RETAIL_CDP_ownpromo_L2[n] + b_DOVE_ALL_BAR_OFFLINE_RETAIL_BBP_RT_ownpromo_L2 * DOVE_ALL_BAR_OFFLINE_RETAIL_BBP_RT_ownpromo_L2[n] + b_DOVE_OTHERS_BAR_OFFLINE_RETAIL_TCP_DM_ownpromo_L2 * DOVE_OTHERS_BAR_OFFLINE_RETAIL_TCP_DM_ownpromo_L2[n] + b_DOVE_OTHERS_BAR_OFFLINE_RETAIL_CPP_ownpromo_L2 * DOVE_OTHERS_BAR_OFFLINE_RETAIL_CPP_ownpromo_L2[n] + b_DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_OOH_haloM_0370 * DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_OOH_haloM_0370[n] + b_DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_PREROLL_haloM_0370 * DOVE_Shampoo_ALL_ALL_ALL_SPENDS_MSP_PREROLL_haloM_0370[n] + b_SET1_RETAIL * SET1_RETAIL[n] + b_SET1_ONLINE * SET1_ONLINE[n], sigma);

}

