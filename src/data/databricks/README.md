# Databricks & Unity Catalog
With the [Databricks Free Edition](https://www.databricks.com/learn/free-edition), I implemented Data Science stacks within my `Databricks Workspace`. 

These processes set up `Unity Catalog` within Databricks environment. These are assets you need to manually created.
 * Create catalog: `nhl-databricks`
  * Create Schema: `data`

This includes data infrastructure and analyses moduels and models.

## Data Process
Data scheduled & pulled through `Github Action` will be used to process and store in catalog.
 * If data are not used for the production purpose, manually upload the formatted table view through 'Data Ingestion' tool in `Databricks`.

## Model Process
All model updates should be updated in `Unity Catelog` as well.