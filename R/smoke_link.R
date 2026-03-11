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
            (mtbs_coverage_perc < x) | !((IDate - 7 < Ig_Date) &
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
    ) %>%
    inner_join(
        treatments,
        by = c("lat", "lon", "year")
    )


severity_agg <- severity %>%
    group_by(event_id) %>%
    summarise(
        mean_severity = mean(dnbr, na.rm = TRUE),
        sum_severity = sum(dnbr, na.rm = TRUE),
        land_type_mode = mode(land_type),
        pixels_no_burned = sum((dnbr <= 0), na.rm = TRUE),
        pixels_low_severity = sum((dnbr > 0 & dnbr < 270), na.rm = TRUE),
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
severity_sample <- severity_agg %>%
    filter(event_id %in%
        (
            severity_agg %>%
                filter(coverage_threshold == 0.9) %>%
                pull(event_id)))


# Traditional regresisons
feols(sum_contrib ~ sum_severity + total_pixels + total_days + coverage_threshold | year,
    data = severity_agg,
    cluster = ~year
) %>%
    summary()

# Save final data
arrow::write_feather(
    severity_agg,
    paste0(save_path, "/severity_emissions_linked_new_inmediate.feather")
)


#################### SEVERITY EXPERIMENTS WITH COEFPLOT ####################
# Filter GEOIDs to only the ones in California (start in 06)
ca_counties <- tigris::counties(state = "CA", cb = TRUE)

pm_county <- read_csv("~/projects/extract/data/smokePM2pt5_predictions_daily_county_20060101-20201231.csv") %>%
    dplyr::filter(GEOID %in% ca_counties$GEOID) %>%
    mutate(year = year(ymd(date))) %>%
    group_by(year) %>%
    summarise(pm25 = sum(smokePM_pred, na.rm = TRUE) / 365)

severity_pixels <- severity %>%
    group_by(year) %>%
    summarise(
        mean_severity = mean(dnbr, na.rm = TRUE),
        sum_severity = sum(dnbr, na.rm = TRUE),
        land_type_mode = mode(land_type),
        pixels_no_burned = sum((dnbr <= 0), na.rm = TRUE),
        pixels_low_severity = sum((dnbr > 0 & dnbr < 270), na.rm = TRUE),
        pixels_mod_low_severity = sum(dnbr < 440 & dnbr > 270, na.rm = TRUE),
        # pixels_mod_high_severity = sum(dnbr < 660 & dnbr > 440, na.rm = TRUE),
        pixels_high_severity = sum((dnbr >= 440), na.rm = TRUE),
        share_low_severity = pixels_low_severity / n(),
        share_high_severity = pixels_high_severity / n(),
        share_no_burned = pixels_no_burned / n(),
        total_pixels = n(),
    ) %>%
    inner_join(pm_county, by = "year")


# Regression
lm(pm25 ~ pixels_low_severity + pixels_mod_low_severity + pixels_high_severity, data = severity_pixels) %>% coefplot(lwdInner = NA, lwdOuter = NA)


lm(pm25 ~ share_no_burned + share_low_severity +
    share_high_severity, data = severity_pixels) %>% summary()
