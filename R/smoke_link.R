# Jeff's amazing matcher function between MTBS and GlobeFire
# This function is used to match MTBS and GlobeFire data
library(pacman)

p_load(
    data.table,
    dplyr,
    purrr,
    tidyr,
    fst,
    sf,
    pbmcapply,
    pbapply,
    latex2exp,
    stringr,
    tigris,
    lubridate,
    arrow,
    duckdb,
    ggplot2,
    estimatr,
    modelsummary,
    fixest,
    tinytable
)


data_path <- "/mnt/sherlock/oak/smoke_linking_public/data"
save_path <- "/mnt/sherlock/oak/prescribed_data/processed/smoke_linking"
data_proc <- "/mnt/sherlock/oak/prescribed_data/processed/"

################################################################################
##################### CLEANING AND MATCHING GLOBFIRE AND MTBS ##################
################################################################################

############################### helper functions ###############################
`%not_in%` <- purrr::negate(`%in%`)

mode <- function(codes) {
    which.max(tabulate(codes))
}

### Aux functions
theme_pset <- function(
    angle = 90,
    axis_text_size = 10,
    axis_title_size = 12,
    title_size = 15,
    legend_text_size = 12,
    strip_text_size = 15,
    legend_position = "bottom") {
    theme_bw(base_size = 12) %+replace%
        theme(
            text = element_text(size = 16, family = "mono"),
            axis.text.x = element_text(
                angle = angle, hjust = NULL, vjust = 0.5
            ),
            panel.background = element_blank(),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            strip.background = element_blank(),
            plot.background = element_rect(fill = "white", colour = NA),
            legend.background = element_rect(fill = "transparent", colour = NA),
            legend.key = element_rect(fill = "transparent", colour = NA),
            axis.text = element_text(size = axis_text_size),
            axis.title = element_text(size = axis_title_size),
            title = element_text(size = title_size),
            legend.text = element_text(size = legend_text_size),
            strip.text = element_text(size = strip_text_size),
            legend.position = legend_position,
            legend.box = "horizontal",
            legend.box.just = "top",
            legend.title.align = 0.5,
        )
}

################################################################################

## us shape
conus_bbox <- sf::st_bbox(c(
    xmin = -127.089844,
    ymin = 22.066785,
    xmax = -66.533203,
    ymax = 50.120578
), crs = st_crs(4326)) %>%
    st_as_sfc()

us_shape <- tigris::nation(resolution = "20m") %>%
    st_transform(crs = st_crs("epsg:4326")) %>%
    st_crop(conus_bbox) %>%
    st_transform(crs = st_crs("epsg:5070"))

# GlobFire and MTBS loading ----

## load globfire data
globfire_df <- st_read(
    paste0(data_path, "/globfire/globfire_na_final_area_2006-2020.shp")
) %>%
    st_transform("epsg:5070") %>%
    st_filter(us_shape) %>%
    mutate(year = year(IDate)) %>%
    st_make_valid()

## load mtbs data
mtbs_df <- read_sf(paste0(data_path, "/mtbs/mtbs_perims_DD.shp")) %>%
    filter(Ig_Date >= "2006-04-19", Ig_Date <= "2020-12-31") %>%
    st_transform("epsg:5070") %>%
    mutate(year = year(Ig_Date)) %>%
    st_make_valid()

# Spatial join by area first filter to relevant years then only keep obs if
# enough area coverage
coverage_threshold <- seq(0.1, 1, 0.1)
year_list <- c(2006:2022)
overlap_list <- lapply(coverage_threshold, function(x) {
    globfire_overlap_df <- pbmclapply(year_list, function(main_year) {
        ## subset data to matching year
        temp_globfire_df <- globfire_df %>%
            filter(year == main_year) %>%
            mutate(burn_area = st_area(geometry))

        temp_mtbs_df <- mtbs_df %>%
            filter(year == main_year) %>%
            mutate(mtbs_burn_area = st_area(geometry))

        ## calculate the area of intersection
        temp_intersection_df <- temp_globfire_df %>%
            st_intersection(temp_mtbs_df) %>%
            mutate(intersection_area = st_area(geometry)) %>%
            dplyr::select(
                Id, Event_ID, irwinID, Incid_Name, Incid_Type, Map_ID,
                Map_Prog, Asmnt_Type, BurnBndAc, BurnBndLat, BurnBndLon,
                Ig_Date, Pre_ID, Post_ID, Perim_ID, mtbs_burn_area,
                intersection_area
            ) %>%
            st_drop_geometry() %>%
            group_by(Id) %>%
            arrange(desc(intersection_area)) %>%
            filter(row_number() == 1)

        ## join to bring in the area of intersection
        temp_globfire_dt <- merge(temp_globfire_df,
            temp_intersection_df,
            by = "Id",
            all.x = TRUE
        ) %>%
            as.data.table()

        # Only keep MTBS data if high area coverage overlap and ignition
        # within start and end globfire date (7 days are added of error).
        subset_cols <- c(
            "Event_ID", "irwinID", "Incid_Name", "Incid_Type", "Map_ID",
            "Map_Prog", "Asmnt_Type", "BurnBndAc", "BurnBndLat", "BurnBndLon",
            "Ig_Date", "Pre_ID", "Post_ID", "Perim_ID"
        )

        # for obs that have low coverage perc and with fire date outside of the
        # globfire start and end window with additional week boundary on start
        # and end set values to NA since its probably not a good match
        temp_globfire_dt[, `:=`(
            coverage_perc = as.numeric(intersection_area / burn_area),
            mtbs_coverage_perc = as.numeric(intersection_area / mtbs_burn_area)
        )][
            (coverage_perc < x) | !((IDate - 7 < Ig_Date) &
                (FDate + 7 > Ig_Date)),
            (subset_cols) := lapply(.SD, function(x) NA),
            .SDcols = subset_cols
        ]
    }) %>%
        bind_rows() %>%
        as.data.frame() %>%
        st_as_sf() %>%
        mutate(coverage_threshold = x)
}) %>%
    bind_rows()

################################################################################
############ FILTERING MTBS AND SUBSETTING ALL SMOKE LINKING FILES #############
################################################################################

# Get fires in CA only using the MTBS Event_ID
mtbs_ca <- overlap_list %>%
    dplyr::filter(!is.na(Event_ID)) %>%
    dplyr::mutate(state = str_sub(Event_ID, 1, 2)) %>%
    dplyr::filter(state %in% c("CA"))

# Load all data
path_to_data <- "/mnt/sherlock/oak/smoke_linking_public/clean/fire_smokepm/30km"
fst_files <- list.files(path_to_data, full.names = TRUE, pattern = ".fst")

# Events in CA only (get all the events per coverage treshold value)
events_by_tresh <- split(mtbs_ca, as.factor(mtbs_ca$coverage_threshold))

data_list_tresh <- lapply(events_by_tresh, function(events) {
    # Process stuff in parallel
    data_list <- pbmclapply(fst_files, function(fst_file) {
        # read in the data
        temp_data <- fst::read_fst(fst_file, as.data.table = TRUE)

        # Filter to only events in MTBS
        temp_data <- temp_data[fire_id %in% unique(events$Id)]
    }, mc.cores = 5) %>%
        bind_rows() %>%
        merge(., events %>%
            dplyr::select(c(Event_ID, Id)) %>%
            mutate(Id = as.character(Id)) %>%
            sf::st_drop_geometry() %>%
            as.data.table(),
        by.x = "fire_id",
        by.y = "Id"
        )

    # Add year and month
    data_list[, `:=`(
        coverage_threshold = unique(events$coverage_threshold),
        year = lubridate::year(date),
        month = lubridate::month(date)
    )]

    return(data_list)
}) %>%
    bind_rows()

################################################################################
##################### AGGREGATE LINKED SMOKE PM BY EVENT_ID/ID #################
################################################################################

# Aggregate data.table by ID
data_agg <- data_list_tresh[, .(
    month_contrib_smokePM = sum(contrib_smokePM, na.rm = TRUE)
), by = c("Event_ID", "coverage_threshold", "date")][
    , .(
        sum_contrib = sum(month_contrib_smokePM, na.rm = TRUE),
        mean_contrib = mean(month_contrib_smokePM, na.rm = TRUE),
        total_days = .N
    ),
    by = c("Event_ID", "coverage_threshold")
]

# Create saving folder if doesn't exist
if (!dir.exists(save_path)) {
    dir.create(save_path, recursive = TRUE)
}
# Write out the data
arrow::write_feather(
    data_agg,
    paste0(save_path, "/smoke_pm_fire_event.feather")
)

data_agg <- arrow::read_feather(
    paste0(save_path, "/smoke_pm_fire_event.feather")
)

################################################################################
######################### BRING SEVERITY IN AND LAND TYPES #####################
################################################################################

# Load treatments to add to severity and intensity plus land type data
# this is pixel level data, so we need to aggregate later
treatments <- arrow::read_feather(
    paste0(data_proc, "/treatments_mtbs.feather")
) %>%
    filter(Event_ID != "nodata" & year >= 2006) %>%
    mutate(Ig_Date = lubridate::date(Ig_Date)) %>%
    select(c(Event_ID, Ig_Date, Incid_Type, year, lat, lon))

# Get event_id from prescribed fires in treatments
rx_fires_id <- treatments %>%
    filter(Incid_Type == "Prescribed Fire") %>%
    pull(Event_ID)

# Load land type data in feather format
land_type <- arrow::read_feather(
    paste0(data_proc, "/land_type/land_type.feather")
)

land_type_event <- land_type %>%
    inner_join(treatments, by = c("lat", "lon")) %>%
    select(c(lat, lon, land_type, Event_ID)) %>%
    group_by(Event_ID) %>%
    summarize(land_type_mode = mode(land_type)) %>%
    mutate(land_type_mode = relevel(as.factor(land_type_mode), ref = "1"))

# Read all parquet files and concatenate them
frp <- arrow::read_feather(
    paste0(data_proc, "/frp_nominal_conf/frp_concat.feather")
) %>%
    select(c(year, frp, grid_id, lat, lon)) %>%
    inner_join(treatments, by = c("year", "lat", "lon")) %>%
    group_by(Event_ID) %>%
    summarise(
        pixels_low_intensity = sum(frp < 100, na.rm = TRUE),
        share_low_intensity = pixels_low_intensity / n(),
        mean_frp = mean(frp, na.rm = TRUE),
        sum_frp = sum(frp, na.rm = TRUE),
        total_pixels = n()
    ) %>%
    inner_join(data_agg, by = c("Event_ID")) %>%
    inner_join(land_type_event, by = "Event_ID")

# Load severity data in feather format
severity <- arrow::read_feather(
    paste0(data_proc, "/dnbr_gee_inmediate/dnbr_long.feather")
) %>%
    mutate(rx = ifelse(event_id %in% rx_fires_id, 1, 0)) %>%
    left_join(
        land_type %>%
            select(c(grid_id, land_type)),
        by = "grid_id"
    ) %>%
    inner_join(mtbs_df %>% select(Event_ID, year) %>% st_drop_geometry(),
        by = c("event_id" = "Event_ID")
    )


severity_agg <- severity %>%
    group_by(event_id) %>%
    summarise(
        mean_severity = mean(dnbr, na.rm = TRUE),
        sum_severity = sum(dnbr, na.rm = TRUE),
        land_type_mode = mode(land_type),
        pixels_low_severity = sum((dnbr < 270), na.rm = TRUE),
        pixels_mod_low_severity = sum(dnbr < 440 & dnbr > 270, na.rm = TRUE),
        pixels_mod_high_severity = sum(dnbr < 660 & dnbr > 440, na.rm = TRUE),
        pixels_high_severity = sum((dnbr >= 660), na.rm = TRUE),
        share_low_severity = pixels_low_severity / n(),
        share_high_severity = pixels_high_severity / n(),
        total_pixels = n(),
        rx_pixels = sum(rx)
    ) %>%
    left_join(mtbs_df %>% select(Event_ID, year) %>% st_drop_geometry(),
        by = c("event_id" = "Event_ID")
    ) %>%
    inner_join(data_agg, by = c("event_id" = "Event_ID")) %>%
    mutate(
        rx_fire = ifelse(event_id %in% rx_fires_id, 1, 0),
        sum_contrib_km = sum_contrib / total_pixels,
        sum_severity_km = sum_severity / total_pixels,
    )
# Get only in-sample severity
severity_sample <- severity %>%
    filter(event_id %in%
        (
            severity_agg %>%
                filter(coverage_threshold == 0.9) %>%
                pull(event_id)))

# Save final data
arrow::write_feather(
    severity_agg,
    paste0(save_path, "/severity_emissions_linked.feather")
)

################################################################################
###################### AGGREGATE SEVERITY AND LAND TYPES #######################
################################################################################

colors <- c(
    "Wildfires (2006-2020)" = "#d95f02",
    "Matched sample" = "#1b9e77"
)

# Histogram of severity
(g <- ggplot(severity, aes(x = dnbr, color = "Wildfires (2006-2020)"))
    +
    geom_density(size = 1)
    +
    geom_density(
        data = severity_sample,
        aes(x = dnbr, color = "Matched sample"), size = 1,
        inherit.aes = FALSE
    )
    +
    scale_color_manual(values = colors)
    +
    labs(
        x = TeX(r"(Severity [ $\Delta NBR$])"),
        y = "Density",
        color = "Legend"
    )
    +
    theme_pset()
)



################################################################################

# Do a plot that shows the relationship between severity and the emissions
severity_agg %>%
    ggplot(aes(x = mean_severity, y = mean_contrib)) +
    geom_smooth(se = TRUE) +
    geom_point(alpha = 0.5) +
    theme_pset()

# Do a plot that shows the distribution of severity by land type
severity_agg %>%
    dplyr::filter(sum_contrib_km > 0) %>%
    ggplot(aes(
        y = sum_contrib_km, x = share_low_severity,
        color = coverage_threshold
    )) +
    geom_smooth(method = "lm", se = TRUE) +
    geom_point(alpha = 0.5) +
    scale_y_continuous(trans = "log10") +
    geom_rug(alpha = 1 / 2, position = "jitter") +
    labs(
        y = "Total emissions per square km",
        x = "Share of Low Severity Fires"
    ) +
    theme_pset(angle = 0)

# Save plot in high-resolution
ggsave(
    paste0("figs/severity_land_type.png"),
    width = 10,
    height = 10,
    dpi = 300
)


################################################################################
#### USE MARISSA'S DATA TO CALCULATE COARSE PM vs. SEVERITY RELATIONSHIP #######
################################################################################

# Load Marissa's data
path_to_data <- "data/misc/tl_2019_us_county/"

county_pop <- read_sf(paste0(path_to_data, "tl_2019_us_county.shp")) %>%
    filter(STATEFP == "06") %>%
    mutate(GEOID = str_c(STATEFP, COUNTYFP))

county_smoke_pm <- readRDS(paste0(path_to_data, "smoke_daily_county.rds"))

# Aggregate data by county/year
county_smoke_pm_agg <- county_smoke_pm %>%
    mutate(state = str_sub(GEOID, 1, 2)) %>%
    filter(state == "06") %>%
    mutate(
        smokePM_pred_coded = ifelse(smokePM_pred < 0, 0, smokePM_pred),
        smokePM_pred_capped = ifelse(smokePM_pred_coded >= 100, 100,
            smokePM_pred_coded
        )
    ) %>%
    inner_join(county_pop, by = "GEOID") %>%
    group_by(year = year(date)) %>%
    summarise(
        mean_pm = mean(smokePM_pred_capped, na.rm = TRUE),
        sum_pm = sum(smokePM_pred_capped, na.rm = TRUE),
        total_days = n()
    )

# Save the data to parquet format
arrow::write_parquet(
    county_smoke_pm_agg,
    paste0(save_path, "/county_smoke_pm.parquet")
)

# Aggregate severity again to get the total emissions per county
smoke_severity_year <- severity %>%
    mutate(rx_pixels = ifelse(event_id %in% rx_fires_id, 1, 0)) %>%
    group_by(year) %>%
    summarise(
        mean_severity = mean(dnbr, na.rm = TRUE),
        median_severity = median(dnbr, na.rm = TRUE),
        sum_severity = sum(dnbr, na.rm = TRUE),
        land_type_mode = mode(land_type),
        pixels_high_severity = sum((dnbr >= 500), na.rm = TRUE),
        pixels_low_severity = sum((dnbr <= 270), na.rm = TRUE),
        share_low_severity = (pixels_low_severity / n()),
        total_pixels = n(),
        number_fires = n_distinct(event_id),
        total_rx_pixels = sum(rx_pixels)
    ) %>%
    inner_join(
        county_smoke_pm_agg,
        by = "year"
    )

# Plot data!
smoke_severity_year %>%
    ggplot(aes(x = mean_severity, y = mean_pm)) +
    geom_smooth(se = TRUE, method = "lm", formula = "y ~ poly(x, 2)") +
    geom_point(alpha = 0.5) +
    theme_pset()

################################################################################
###################### REGRESSIONS OF SMOKE PM ON SEVERITY #####################
################################################################################

run_reg <- function(
    y,
    x,
    controls,
    order,
    data,
    split_train = 0.8,
    fe = FALSE,
    remove_outliers = TRUE) {
    x_sym <- rlang::sym(x)

    # Remove outliers
    if (remove_outliers == 1) {
        print("Removing outliers")
        data <- data %>%
            dplyr::filter(
                !!x_sym >= quantile(!!x_sym, 0.05) &
                    !!x_sym <= quantile(!!x_sym, 0.95)
            )
    }

    # Sample data in train/test splits
    set.seed(42)
    train_idx <- sample(seq_len(nrow(data)), split_train * nrow(data))
    train_data <- data[train_idx, ]
    test_data <- data[-train_idx, ]


    # Build formula using controls and dv
    if (isTRUE(fe)) {
        formula <- paste0(
            y, " ~ ", glue::glue("poly({x}, {order}, raw=TRUE)"), "+",
            paste0(controls, collapse = " + "), "  | year "
        )
        mod <- fixest::feols(as.formula(formula), data = train_data)
        r2 <- r2(mod, "war2")
    } else {
        formula <- paste0(
            y, " ~ ", glue::glue("poly({x}, {order}, raw=TRUE)"), "+",
            paste0(controls, collapse = " + ")
        )
        mod <- lm(as.formula(formula), data = train_data)
        r2 <- 0
    }

    # Calculate RMSE
    y_vec <- train_data[y] |> dplyr::pull()
    rmse <- sqrt(sum((fitted(mod) - y_vec)^2) / length(y_vec))

    # Extract model RMSE
    mod_diag <- broom::glance(mod) |>
        dplyr::mutate(rmse_train = rmse, order = order)

    # Run predictions in test data
    preds <- predict(mod, newdata = test_data)

    y_vec <- test_data[y] |> dplyr::pull()
    rmse <- sqrt(sum((preds - y_vec)^2) / length(y_vec))
    # Add test RMSE
    mod_diag <- mod_diag |>
        dplyr::mutate(rmse_test = rmse, r2 = r2) |>
        dplyr::select(c(rmse_train, rmse_test, r2, order, adj.r.squared))

    # Run model with full data
    if (isTRUE(fe)) {
        mod_full <- fixest::feols(as.formula(formula), data = data)
    } else {
        mod_full <- lm(as.formula(formula), data = data)
    }

    return(
        list(
            mod = mod_full, out = mod_diag,
            remove_outliers = remove_outliers,
            fe = fe, data = data
        )
    )
}


# Build grid to search for best model
grid <- expand.grid(
    deg_ord = 1:4,
    thresh = seq(0.5, 0.9, 0.1)
)

# Apply function to grid
grid_out <- pbmclapply(seq_along(1:nrow(grid)), function(i) {
    # Loop data
    data <- severity_agg %>%
        dplyr::filter(coverage_threshold == grid$thresh[i])

    if (nrow(data) == 0) {
        return(NULL)
    } else {
        out <- run_reg(
            y = "sum_contrib",
            x = "sum_severity",
            controls = c("total_days", "total_pixels"),
            order = grid$deg_ord[i],
            data = data,
            fe = TRUE,
            split_train = 0.8,
            remove_outliers = TRUE
        )

        df <- out$out %>%
            mutate(
                thresh = grid$thresh[i],
                mod = list(out$mod),
                data = list(out$data)
            )
    }
}) %>%
    bind_rows() %>%
    group_by(thresh) %>%
    mutate(rank_rmse = rank(rmse_test)) %>%
    filter(rank_rmse == 1)


create_predict_dataset <- function(df, mod) {
    data.frame(
        sum_severity = seq(
            min(df$sum_severity),
            max(df$sum_severity),
            length.out = 100
        ),
        total_days = mean(df$total_days),
        total_pixels = mean(df$total_pixels),
        year = 2020
    ) %>% dplyr::mutate(preds = predict(mod, newdata = .))
}

calculate_marginal_effects <- function(mod) {
    marginaleffects::slopes(mod,
        variables = "sum_severity"
    )
}


grid_preds <- grid_out %>%
    mutate(preds = map2(data, mod, create_predict_dataset)) %>%
    dplyr::select(thresh, preds) %>%
    tidyr::unnest(preds)

grid_data <- grid_out %>%
    dplyr::select(thresh, data) %>%
    tidyr::unnest(data)

grid_margins <- grid_out %>%
    dplyr::select(thresh, mod) %>%
    dplyr::mutate(margins = map(mod, calculate_marginal_effects)) %>%
    dplyr::select(thresh, margins) %>%
    tidyr::unnest(margins)

g <- (
    ggplot(
        data = grid_data,
        aes(
            x = sum_severity,
            y = sum_contrib,
            color = as.factor(thresh)
        )
    ) +
        geom_point(alpha = 0.5) +
        geom_line(
            data = grid_preds,
            aes(
                x = sum_severity,
                y = preds,
                color = as.factor(thresh)
            ),
            inherit.aes = FALSE
        ) +
        facet_wrap(
            vars(thresh),
            scales = "free_y",
            nrow = 5
        ) +
        scale_color_manual(
            name = "Coverage Threshold",
            values = c(
                "#7fc97f",
                "#beaed4",
                "#fdc086",
                "#386cb0",
                "#f0027f"
            )
        ) +
        labs(
            x = TeX(r"(Sum Severity Matched Fires [ $\Delta NBR$])"),
            y = TeX(r"(Total Matched Fires  [ PM$_{2.5}$])")
        ) +
        theme_bw()
)


# Plot marginal values
g_m <- (
    ggplot() +
        geom_point(alpha = 0.5) +
        geom_line(
            data = grid_margins,
            aes(
                x = sum_severity,
                y = estimate,
                color = as.factor(thresh)
            ),
            inherit.aes = FALSE
        ) +
        geom_ribbon(
            data = grid_margins,
            aes(
                x = sum_severity,
                ymin = conf.low,
                ymax = conf.high,
            ),
            inherit.aes = FALSE,
            alpha = 0.2
        ) +
        facet_wrap(
            vars(thresh),
            scales = "fixed",
            nrow = 5
        ) +
        scale_color_manual(
            name = "Coverage Threshold",
            values = c(
                "#7fc97f",
                "#beaed4",
                "#fdc086",
                "#386cb0",
                "#f0027f"
            )
        ) +
        labs(
            x = TeX(r"(Sum Severity Matched Fires [ $\Delta NBR$])"),
            y = TeX(r"(Total Matched Fires  [ PM$_{2.5}$])")
        ) +
        theme_bw()
)


# Save plot
ggsave(
    paste0("severity_pm_grid.png"),
    width = 10,
    height = 10,
    dpi = 300
)


gm <- tibble::tribble(
    ~raw, ~clean, ~fmt,
    "nobs", "N", 0,
    "adj.r.squared", "Adj R^{2}", 3,
    "r2.within.adjusted", "Within R^{2}", 3,
)

add_rows <- tibble::tribble(
    ~term, ~model, ~model, ~model, ~model, ~model,
    "Year FE", "no", "no", "yes", "yes", "yes",
    "Land Type FE", "no", "no", "no", "yes", "Conifers"
)

models <- list(
    lm_robust(sum_pm ~ sum_severity + year, data = smoke_severity_year),
    lm_robust(
        sum_contrib ~ sum_severity,
        data = severity_agg %>%
            filter((coverage_threshold == 0.6))
    ),
    fixest::feols(sum_contrib ~ sum_severity | year,
        data = severity_agg %>%
            filter((coverage_threshold == 0.6))
    ),
    fixest::feols(sum_contrib ~ sum_severity | year + land_type_mode,
        data = severity_agg %>%
            filter((coverage_threshold == 0.6))
    ),
    fixest::feols(
        sum_contrib ~ pixels_low_severity + pixels_mod_low_severity +
            pixels_mod_high_severity + pixels_high_severity | year,
        data = severity_agg %>%
            filter((coverage_threshold == 0.6))
    )
)


modelsummary(models,
    estimate = "{estimate}{stars}",
    coef_rename = c(
        sum_severity = "Severity",
        mean_severity = "Severity",
        sum_contrib_km = "Severity",
        sum_pm = "Total PM",
        mean_pm = "Mean PM",
        sum_contrib = "Total PM",
        mean_contrib = "Mean PM"
    ),
    # add_rows = add_rows,
    coef_omit = c("(Intercept)"),
    # gof_map = gm,
    # output = "latex"
)
#|> group_tt(j = list(State = 2:2, Fires = 2:6))
