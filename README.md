# ADS Final Report

For the fynesse template and the final report notebook.

## Data Source

There are mainly three data sources used in this project:

- [UK House Price Data](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)
  - Contains HM Land Registry data © Crown copyright and database right 2021. This data is licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). Price Paid Data contains address data processed against Ordnance Survey’s AddressBase Premium product, which incorporates Royal Mail’s PAF® database (Address Data).

- [Open Postcode Geo](https://www.getthedata.com/open-postcode-geo)
    - Open Postcode Geo is derived from the ONS Postcode Directory which is licenced under the Open Government Licence and the Ordnance Survey OpenData Licence. See the [ONS Licences](https://www.ons.gov.uk/methodology/geography/licences) page for more information.

- [OpenStreetMap](https://openstreetmap.org/copyright)
    - All the OSM data is available under the Open Database License. The above link has information about OpenStreetMap’s data sources as well as the ODbL.

## Get started

- Simply use the assessment notebook located at ```notebooks/ads_course_assessment.ipynb```. Follow the instructions in the notebook. Also refer to the next section for some general idea of the pipeline.

- Tests can be run using ```pytest``` from the project root. It is also recommended to disable slow tests using ```pytest -m "not slow_for_db and not slow_locally"```

## Pipeline

The fynesse template is used in this assessment. The notebook is the core running centre, while some tests are available in the test directory for doing some of the more dirty checks.

- Access stage
  - In this stage we are populating database to get ready for analysis
  - For each database, we will go through creation, local data download, upload, and then final setup (indexing rows)
  - The local data will be downloaded at a tenporary folder ```tmp_data``` in the same folder as the notebook.
  - Keep in mind that the whole process could take a long time, so it is recommended not to run it again unless the data is lost.
  - For legal attributions, relevant information could be found in either the Data Source section or the Attribution section in the notebook.

- Assess stage
  - First I did some sanity checks on the data from the database. The tests are put in the test folder since they could take quite some time to complete, and they usually don't show anything interesting.
  - Then I did some general visualisation on some sampled data from the database. The reason for sampling is to avoid fetching all the data, which could take ~30 minutes on its on. The number of sampled data could be controled in the ```defaults.yml``` file.
  - There are also some "helper" methods in the module, for example a method that one-hot encodes a column, and a method that retreives places of interest from the open street map API. These methods are partially used in the visualisation, and they certainly help reduce the complexity of the last stage.

- Address stage
  - The final step - actually predicting the price given some location and relevant information.
  - I followed the "intended" method in the notebook - that is, use data around the given location and date, train the model on the fly, then validate the model appropriately. Everything about prediction is encapsulated in the ```predict_price``` method, and information about the accuracy (as $R^2$ value from CV) or relevant visualisation could be showed by adjusting the ```validation_level``` parameter. Note that the process is going to take a long time (usually around 10 minutes per prediction) due to the training data being prepared and processed on the fly. This could be mitigated by selecting a smaller size of bounding box, smaller date range, or disabling the place of interest processing.