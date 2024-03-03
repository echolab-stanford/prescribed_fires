# Load all libraries for analysis!
pacman::p_load(fixest, arrow, tidyverse, data.table,
               did, bacondecomp, panelView)
pacman::p_load_gh("xuyiqing/paneltools")

# Read in data from arrow file from Python
df <- read_feather(
    "/mnt/sherlock/oak/prescribed_data/processed/wildfires_panel.feather"
    )

# Create variables for panel analysis manually
df_filter <- df %>%
    mutate(
        min_treat_year = ifelse(is.na(min_treat_year), 10000, min_treat_year),
        rel_year = ifelse(is.na(rel_year), 0, rel_year),
        low_intensity = ifelse(first_frp <= 2, 1, 0),
        low_severity = ifelse(first_dnbr <= 3, 1, 0),
        treat_severity = low_intensity * treat,
        treat_intensity = low_severity * treat
    )

df_intensity <- df_filter %>%
    group_by(grid_id) %>%
    mutate(treat_intensity_mod = ifelse(rel_year > 0, NA, treat_intensity)) %>%
    tidyr::fill(treat_intensity_mod, .direction = "down") %>%
    paneltools::get.cohort(., D = "treat_intensity_mod",
                            index = c("grid_id", "year")) %>%
    mutate(Time_to_Treatment = ifelse(is.na(Time_to_Treatment),
                             0, Time_to_Treatment),
            FirstTreat = ifelse(is.na(FirstTreat), 1000, FirstTreat)
    )

df_severity <- df_filter %>%
    group_by(grid_id) %>%
    mutate(treat_severity_mod = ifelse(rel_year > 0, NA, treat_severity)) %>%
    tidyr::fill(treat_severity_mod, .direction = "down") %>%
    paneltools::get.cohort(., D = "treat_severity_mod",
                            index = c("grid_id", "year")) %>%
    mutate(Time_to_Treatment = ifelse(is.na(Time_to_Treatment),
                             0, Time_to_Treatment),
            FirstTreat = ifelse(is.na(FirstTreat), 1000, FirstTreat)
    )

################################################################################
########################## SUN & ABRAHAMS WEIGHTS ##############################
################################################################################

# Calculate Sun and Abraham (2020) full-dyamics weights (Eq. 12):
# Create combos of relative dates with min_treat_year
grid_weights <- expand.grid(
    cohort = unique(df_filter$min_treat_year),
    relative_date = 2
)

weights_df <- mapply(function(cohort, relative_date) {
    # Calculate CATE LHS
    cate_ols <- df_filter %>%
    mutate(cate = ifelse(min_treat_year == cohort &
    rel_year == relative_date, 1, 0))

    # Check that there are enough observations
    if (sum(cate_ols$cate) > 0) {

        # Calculate regression weights
        print(paste0("Cohort", cohort, " == Relative Date", relative_date))
        reg <- feols(cate ~ i(rel_year, ref = -1) | grid_id + year,
                     data = cate_ols %>% filter(count_treats %in% c(1:2))
        )

        # Extract weights
        weights <- broom::tidy(reg, conf.int = 0.95) %>%
        mutate(cohort = cohort,
            relative_date = relative_date)
        return(weights)
    } else {
        return(NULL)
    }
}, cohort = grid_weights[, "cohort"],
relative_date = grid_weights[, "relative_date"],
 SIMPLIFY = FALSE)

# Bind all weights
weights_df_concat <- bind_rows(weights_df) %>%
    separate(term, sep = ":", into = c(NA, NA, "rel_year"))

# Plot weights
g <- (
    ggplot(weights_df_concat, aes(
        x = as.integer(rel_year),
        y = estimate,
        color = cohort)
    )
        +
        geom_point()
        +
        geom_hline(yintercept = 0, linetype = "dashed", color = "gray50")
        +
        geom_pointrange(aes(ymin = conf.low, ymax = conf.high))
        +
        # Change scale to only include first and last element of the cohort
        scale_color_continuous(name = "Cohort", breaks = c(2000, 2019))
        +
        labs(y = "Weights", x = "Time from first fire [years]")
        +
        theme_bw()
        +
        theme(legend.position = "bottom")
        +
        theme(text = element_text(size = 20))
        )
ggsave("figs/weights_rel_2.png", g, width = 10, height = 10, units = "in")


################################################################################
############################ TESTING FEOLS #####################################
################################################################################

# Standard regression!
naive <- feols(
        frp ~ i(Time_to_Treatment, treat_intensity_mod, ref = -1) |
        grid_id + year,
    data = df_intensity %>%
    dplyr::filter(count_treats %in% c(1, 2))
)

# Test Sun and Abrahams regression
sunab <- feols(frp ~ sunab(FirstTreat, year) |
    grid_id + year,
    data = df_intensity %>%
    dplyr::filter(count_treats %in% c(1, 2))
    )

cs <- att_gt(yname = "frp",
                 gname = "FirstTreat",
                 idname = "grid_id",
                 tname = "year",
                 xformla = ~1,
                 control_group = "nevertreated",
                 allow_unbalanced_panel = TRUE,
                 data = df_intensity %>%
                 dplyr::filter(count_treats %in% c(1, 2)),
                 est_method = "reg")

cs_event <- aggte(cs, type = "dynamic",
                  bstrap = FALSE, cband = FALSE, na.rm = T)

iplot(list(naive, sunab), ref = "all", ref.line = 1)

################################################################################
############################ INTENSITY FEOLS ###################################
################################################################################

# Run panel models with feols by land type
intensity_feols <- feols(
    frp ~ i(Time_to_Treatment, treat_intensity_mod, ref = -1) |
    grid_id + year,
    data = df_intensity %>%
    dplyr::filter(count_treats %in% c(1, 2)))

intensity_feols_conifer <- feols(
    frp ~ i(Time_to_Treatment, treat_intensity_mod, ref = -1) |
    grid_id + year,
    data = df_intensity %>%
    dplyr::filter(count_treats %in% c(1, 2)) %>%
    filter(land_type %in% c(2, 3)))

intensity_feols_shurbs <- feols(
    frp ~ i(Time_to_Treatment, treat_intensity_mod, ref = -1) |
    grid_id + year,
    data = df_intensity %>%
    dplyr::filter(count_treats %in% c(1, 2)) %>%
    filter(land_type %in% c(12, 13)))

# Put all models in list
models <- list(
    intensity_feols,
    intensity_feols_conifer,
    intensity_feols_shurbs
    )

# Transform models to tidy data frame and add land type
df_coefs <- mapply(function(x, y) {
    broom::tidy(x, conf.int = 0.95) %>%
        separate(term,
            sep = ":",
            into = c(NA, NA, "period", NA)
        ) %>%
        mutate(
            period = as.integer(period),
            land_type = y
        )
}, x = models, y = c("All", "Conifer", "Shrub"), SIMPLIFY = FALSE) %>%
    bind_rows()

# Plots severity!
 g <- (
      ggplot(df_coefs, aes(
          x = as.integer(period),
          y = estimate,
          color = land_type)
      )
          +
          geom_point()
          +
          geom_hline(yintercept = 0, linetype = "dashed", color = "gray50")
          +
          geom_pointrange(aes(ymin = conf.low, ymax = conf.high))
          +
          scale_color_discrete(name = "Land Type")
          +
          labs(y = "Estimate [FRP]", x = "Time from first fire [years]")
          +
          theme_bw()
          +
          theme(legend.position = "bottom")
          +
          theme(text = element_text(size = 20))
          )
# Save plot
ggsave("figs/intensity_test_feols.png", g, width = 10, height = 10, units = "in")

######################################################################
############################ SEVERITY FEOLS ##########################
######################################################################

# Run panel models with feols by land type
severity_feols <- feols(
    dnbr ~ i(Time_to_Treatment, treat_severity_mod, ref = -1) |
    grid_id + year,
    data = df_severity %>%
    dplyr::filter(count_treats %in% c(1, 2))
)

severity_feols_conifer <- feols(
    dnbr ~ i(Time_to_Treatment, treat_severity_mod, ref = -1) |
    grid_id + year,
    data = df_severity %>%
    dplyr::filter(count_treats %in% c(1, 2)) %>%
     filter(land_type %in% c(2, 3))
)

severity_feols_shurbs <- feols(
    dnbr ~ i(Time_to_Treatment, treat_severity_mod, ref = -1) |
    grid_id + year,
    data = df_severity %>%
    dplyr::filter(count_treats %in% c(1, 2)) %>%
     filter(land_type %in% c(12, 13))
)

# Put all models in list
models <- list(
    severity_feols,
    severity_feols_conifer,
    severity_feols_shurbs
    )

# Transform models to tidy data frame and add land type
df_coefs <- mapply(function(x, y) {
    broom::tidy(x, conf.int = 0.95) %>%
        separate(term,
            sep = ":",
            into = c(NA, NA, "period", NA)
        ) %>%
        mutate(
            period = as.integer(period),
            land_type = y
        )
}, x = models, y = c("All", "Conifer", "Shrub"), SIMPLIFY = FALSE) %>%
    bind_rows()


# Plots severity!
 g <- (
      ggplot(df_coefs, aes(
          x = as.integer(period),
          y = estimate,
          color = land_type)
      )
          +
          geom_point()
          +
          geom_hline(yintercept = 0, linetype = "dashed", color = "gray50")
          +
          geom_pointrange(aes(ymin = conf.low, ymax = conf.high))
          +
          scale_color_discrete(name = "Land Type")
          +
          labs(y = "Estimate [dNBR]", x = "Time from first fire [years]")
          +
          theme_bw()
          +
          theme(legend.position = "bottom")
          +
          theme(text = element_text(size = 20))
          )
# Save plot
ggsave("figs/severity_feols.png", g, width = 10, height = 10, units = "in")
