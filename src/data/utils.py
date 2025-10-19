from ..submodule.jp_qcew.src.data.data_process import cleanData
from ..submodule.aea.src.data_pull import DataPull
import polars as pl
import os
import pandas as pd
import logging
import geopandas as gpd
from shapely import wkt


class DiffReg(cleanData, DataPull):
    def __init__(
        self,
        saving_dir: str = "data/",
        database_file: str = "data.ddb",
        log_file: str = "data_process.log",
    ):
        super().__init__(saving_dir, database_file, log_file)

    def base_data(self) -> pl.DataFrame:
        if "qcewtable" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            self.make_qcew_dataset()

        df_qcew = self.conn.sql(
            "SELECT year,qtr,phys_addr_5_zip,ui_addr_5_zip,mail_addr_5_zip,ein,first_month_employment,total_wages,second_month_employment,third_month_employment,naics_code FROM qcewtable"
        ).pl()
        df_qcew = df_qcew.rename({"phys_addr_5_zip": "zipcode"})
        df_qcew = df_qcew.filter(
            (pl.col("zipcode") != "") & (pl.col("naics_code") != "")
        )
        df_qcew = df_qcew.with_columns(
            first_month_employment=pl.col("first_month_employment").fill_null(
                strategy="zero"
            ),
            second_month_employment=pl.col("second_month_employment").fill_null(
                strategy="zero"
            ),
            third_month_employment=pl.col("third_month_employment").fill_null(
                strategy="zero"
            ),
            total_wages=pl.col("total_wages").fill_null(strategy="zero"),
        )
        df_qcew = df_qcew.with_columns(
            total_employment=(
                pl.col("first_month_employment")
                + pl.col("second_month_employment")
                + pl.col("third_month_employment")
            )
            / 3
        )
        df_qcew = df_qcew.filter(
            (pl.col("total_employment") != 0) & (pl.col("total_wages") != 0)
        )
        df_qcew = df_qcew.with_columns(
            wages_employee=pl.col("total_wages") / pl.col("total_employment"),
            sector=pl.col("naics_code").str.slice(0, 2),
        )
        return df_qcew

    def make_zips_table(self) -> pd.DataFrame:
        # initiiate the database tables
        if "zipstable" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            # Download the shape files
            if not os.path.exists(f"{self.saving_dir}external/zips_shape.zip"):
                self.pull_file(
                    url="https://www2.census.gov/geo/tiger/TIGER2024/ZCTA520/tl_2024_us_zcta520.zip",
                    filename=f"{self.saving_dir}external/zips_shape.zip",
                )
                logging.info("Downloaded zipcode shape files")

            # Process and insert the shape files
            gdf = gpd.read_file(f"{self.saving_dir}external/zips_shape.zip")
            gdf = gdf[gdf["ZCTA5CE20"].str.startswith("00")]
            gdf = gdf.rename(columns={"ZCTA5CE20": "zipcode"}).reset_index()
            gdf = gdf[["zipcode", "geometry"]]
            gdf["zipcode"] = gdf["zipcode"].str.strip()
            df = gdf.drop(columns="geometry")
            geometry = gdf["geometry"].apply(lambda geom: geom.wkt)
            df["geometry"] = geometry
            self.conn.execute("CREATE TABLE zipstable AS SELECT * FROM df")
            logging.info(
                f"The zipstable is empty inserting {self.saving_dir}external/cousub.zip"
            )
        return self.conn.sql("SELECT * FROM zipstable;").df()

    def spatial_data(self) -> gpd.GeoDataFrame:
        gdf_zips = gpd.GeoDataFrame(self.make_zips_table())
        gdf_zips["geometry"] = gdf_zips["geometry"].apply(wkt.loads)
        gdf_zips = gdf_zips.set_geometry("geometry").set_crs(
            "EPSG:4269", allow_override=True
        )
        gdf_zips = gdf_zips.to_crs("EPSG:3395")
        gdf_zips["zipcode"] = gdf_zips["zipcode"].astype(str)

        gdf_county = self.pull_county_shapes()

        gdf = gpd.sjoin(
            gdf_zips,
            gdf_county[["geometry", "area_fips"]],
            how="inner",
            predicate="intersects",
        ).drop("index_right", axis=1)

        return gdf
