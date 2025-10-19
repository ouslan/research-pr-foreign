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
            how="left",
            predicate="intersects",
        ).drop("index_right", axis=1)

        gdf = gdf.drop_duplicates(subset=["zipcode"])

        return gdf

    def regular_data(self, naics: str, foreign: bool):
        df_qcew = self.base_data().filter(pl.col("year") >= 2012)
        if naics == "31-33":
            df_qcew = df_qcew.filter(
                (pl.col("naics_code").str.starts_with("31"))
                | (pl.col("naics_code").str.starts_with("32"))
                | (pl.col("naics_code").str.starts_with("33"))
            )
        elif naics == "44-45":
            df_qcew = df_qcew.filter(
                (pl.col("naics_code").str.starts_with("44"))
                | (pl.col("naics_code").str.starts_with("45"))
            )
        elif naics == "48-49":
            df_qcew = df_qcew.filter(
                (pl.col("naics_code").str.starts_with("48"))
                | (pl.col("naics_code").str.starts_with("49"))
            )
        elif naics == "72-accommodation":
            df_qcew = df_qcew.filter(
                (pl.col("naics_code").str.starts_with("7211"))
                | (pl.col("naics_code").str.starts_with("7212"))
                | (pl.col("naics_code").str.starts_with("7213"))
            )
        elif naics == "72-food":
            df_qcew = df_qcew.filter(
                (pl.col("naics_code").str.starts_with("7223"))
                | (pl.col("naics_code").str.starts_with("7224"))
                | (pl.col("naics_code").str.starts_with("7225"))
            )
        else:
            df_qcew = df_qcew.filter(pl.col("naics_code").str.starts_with(naics))

        df_qcew = df_qcew.filter(pl.col("ein") != "")
        pr_zips = self.spatial_data()["zipcode"].to_list()

        df_qcew = df_qcew.with_columns(
            foreign=pl.when(pl.col("ui_addr_5_zip").is_in(pr_zips)).then(0).otherwise(1)
        )

        if foreign:
            df_qcew = df_qcew.filter(pl.col("foreign") == 1)
        else:
            df_qcew = df_qcew.filter(pl.col("foreign") == 0)

        df_zips = pl.DataFrame(self.spatial_data().drop("geometry", axis=1))

        df_qcew = df_qcew.join(df_zips, on=["zipcode"], how="inner", validate="m:1")

        df_qcew = df_qcew.group_by(["year", "qtr", "area_fips"]).agg(
            total_employment=pl.col("total_employment").sum(),
            total_wages=pl.col("total_wages").sum(),
        )

        df_qcew = df_qcew.with_columns(
            avg_wkly_wage=pl.col("total_wages") / (pl.col("total_employment") * 13),
            min_wage=pl.when((pl.col("year") >= 2002) & (pl.col("year") < 2010))
            .then(5.15 * 40)
            .when((pl.col("year") >= 2010) & (pl.col("year") < 2023))
            .then(7.25 * 40)
            .when(pl.col("year") == 2023)
            .then(8.5 * 40)
            .when(pl.col("year") == 2024)
            .then(10.5 * 40)
            .otherwise(-1),
        )
        df_qcew = df_qcew.with_columns(
            k_index=pl.col("min_wage") / pl.col("avg_wkly_wage")
        )

        data = df_qcew.to_pandas().copy()

        data = data.sort_values(["year", "qtr", "area_fips"]).reset_index(drop=True)

        return data
