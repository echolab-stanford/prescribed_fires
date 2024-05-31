# Jeff's amazing matcher function between MTBS and GlobeFire
# This function is used to match MTBS and GlobeFire data

library(data.table)
library(dplyr)
library(fst)
library(sf)
library(pbmcapply)
library(pbapply)
library(stringr)
library(tigris)
library(lubridate)
library(arrow)
library(ggplot2)
library(estimatr)


data_path <- "/mnt/sherlock/oak/smoke_linking_public/data"
save_path <- "/mnt/sherlock/oak/prescribed_data/processed/smoke_linking"
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
coverage_threshold <- 0.2
year_list <- c(2006:2020)
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

    # Only keep MTBS data if high area coverage overlap and ignition date
    # within start and end globfire date
    subset_cols <- c(
        "Event_ID", "irwinID", "Incid_Name", "Incid_Type", "Map_ID",
        "Map_Prog", "Asmnt_Type", "BurnBndAc", "BurnBndLat", "BurnBndLon",
        "Ig_Date", "Pre_ID", "Post_ID", "Perim_ID"
    )

    # for obs that have low coverage perc and with fire date outside of the
    # globfire start and end window with additional week boundary on start and
    # end set values to NA since its probably not a good match

    temp_globfire_dt[, `:=`(
        coverage_perc = as.numeric(intersection_area / burn_area),
        mtbs_coverage_perc = as.numeric(intersection_area / mtbs_burn_area)
    )][
        (coverage_perc < coverage_threshold) | !((IDate - 7 < Ig_Date) &
            (FDate + 7 > Ig_Date)),
        (subset_cols) := lapply(.SD, function(x) NA),
        .SDcols = subset_cols
    ]
}) %>%
    bind_rows() %>%
    as.data.frame() %>%
    st_as_sf()

################################################################################
############ FILTERING MTBS AND SUBSETTING ALL SMOKE LINKING FILES #############
################################################################################

# Get fires in CA only using the MTBS Event_ID
mtbs_ca <- globfire_overlap_df %>%
    dplyr::filter(!is.na(Event_ID)) %>%
    dplyr::mutate(state = str_sub(Event_ID, 1, 2)) %>%
    dplyr::filter(state %in% c("CA"))

# Events in CA only
events <- mtbs_ca %>% pull(Id)

# Load all data
path_to_data <- "/mnt/sherlock/oak/smoke_linking_public/clean/fire_smokepm/30km"
fst_files <- list.files(path_to_data, full.names = TRUE, pattern = ".fst")

# Process stuff in parallel
data_list <- pbmclapply(fst_files, function(fst_file) {
    ## read in the data
    temp_data <- fst::read_fst(fst_file, as.data.table = TRUE)

    # Filter to only events in MTBS
    temp_data <- temp_data[fire_id %in% events]

    ## return the data
    return(temp_data)
}, mc.cores = 5) %>%
    bind_rows() %>%
    merge(., mtbs_ca %>%
        dplyr::select(c(Event_ID, Id)) %>%
        mutate(Id = as.character(Id)) %>%
        sf::st_drop_geometry() %>%
        as.data.table(),
    by.x = "fire_id",
    by.y = "Id"
    )

data_list[, `:=`(year = lubridate::year(date), month = lubridate::month(date))]

################################################################################
##################### AGGREGATE LINKED SMOKE PM BY EVENT_ID/ID #################
################################################################################

# Aggregate data.table by ID
data_agg <- data_list[, .(
    mean_contrib = mean(contrib_smokePM, na.rm = TRUE),
    sum_contrib = sum(contrib_smokePM, na.rm = TRUE)
), by = "Event_ID"]

# Create saving folder if doesn't exist
if (!dir.exists(save_path)) {
    dir.create(save_path, recursive = TRUE)
}
# Write out the data
arrow::write_feather(
    data_agg,
    paste0(save_path, "/smoke_pm_fire_event.feather")
)

################################################################################
######################### BRING SEVERITY IN AND LAND TYPES #####################
################################################################################

data_proc <- "/mnt/sherlock/oak/prescribed_data/processed/"

# Load treatments to add to severity
treatments <- arrow::read_feather(
    paste0(data_proc, "/treatments_mtbs.feather")
) %>%
    filter(Event_ID != "nodata") %>%
    mutate(Ig_Date = lubridate::date(Ig_Date)) %>%
    select(c(Event_ID, Ig_Date, Incid_Type, year, lat, lon))

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
frp_files <- list.files(
    paste0(data_proc, "frp/frp_parquet"),
    pattern = ".parquet",
    full.names = TRUE
)

frp <- lapply(frp_files, arrow::read_parquet) %>%
    bind_rows() %>%
    select(c(time, frp, grid_id, lat, lon)) %>%
    mutate(
        time = lubridate::date(time),
        month = month(time),
        year = year(time)
    ) %>%
    inner_join(treatments, by = c("year", "lat", "lon")) %>%
    group_by(Event_ID) %>%
    summarise(
        mean_frp = mean(frp, na.rm = TRUE),
        sum_frp = sum(frp, na.rm = TRUE)
    ) %>%
    inner_join(data_agg, by = c("Event_ID")) %>%
    inner_join(land_type_event, by = "Event_ID")

# Load severity data in feather format
severity <- arrow::read_feather(
    paste0(data_proc, "/dnbr_gee/dnbr_long.feather")
) %>%
    left_join(land_type, by = "grid_id") %>%
    mutate(dnbr = ifelse(dnbr <= 0, NA, dnbr))

severity_agg <- severity %>%
    group_by(event_id) %>%
    summarise(
        mean_severity = mean(dnbr, na.rm = TRUE),
        land_type_mode = mode(land_type),
        pixels_low_severity = sum((dnbr <= 250) & (dnbr >= 100), na.rm = TRUE),
        share_low_severity = pixels_low_severity / n()
    ) %>%
    inner_join(
        mtbs_ca %>%
            select(Event_ID, mtbs_burn_area) %>%
            st_drop_geometry(),
        by = c("event_id" = "Event_ID")
    ) %>%
    mutate(
        mtbs_burn_area = as.numeric(mtbs_burn_area) * 1e-6,
        share_low_severity_mtbs = pixels_low_severity / mtbs_burn_area,
        share_low_severity_mtbs = ifelse(
            share_low_severity >= 1, NA, share_low_severity
        )
    ) %>%
    inner_join(data_agg, by = c("event_id" = "Event_ID"))

# Do a plot that shows the distribution of severity by land type
severity_agg %>%
    filter(land_type_mode %in% c(2, 12)) %>%
    ggplot(aes(
        y = sum_contrib, x = share_low_severity_mtbs,
        color = as.factor(land_type_mode)
    )) +
    geom_smooth(method = "lm", se = TRUE) +
    geom_point(alpha = 0.5) +
    theme_bw() +
    scale_y_log10() +
    labs(
        y = "Sum emissions",
        x = "Share of Low Severity Fires"
    )
ggplot2::ggsave(
    paste0(save_path, "fire_severity_land_type.png"),
    width = 6,
    height = 6
)

lm_robust(sum_contrib ~ sum_frp * as.factor(land_type_mode),
    data = frp %>% filter(land_type_mode %in% c(2, 12)),
) %>%
    summary()

lm_robust(sum_contrib ~ share_low_severity * as.factor(land_type_mode),
    data = severity_agg %>% filter(land_type_mode %in% c(2, 12)),
) %>%
    summary()
