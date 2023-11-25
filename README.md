# ADS Final Report

For the fynesse template and the final report notebook.

## TODO:

- access: tests
- assess: sanity check
- assess: general PCA & visualisation
- assess: tests
- address: summarise decisions and reasonings
- address: feature transform
- address: validation PCA
- address: cross-validation
- address: predict examples
- address: tests
- address: llm documentation
- notebook transferring
- code documentation
- general documentation
- code refactoring

## Notes:

- run tests using ```pytest```

## Data Source

There are mainly three data sources used in this project:

- [UK House Price Data](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)
  - Contains HM Land Registry data © Crown copyright and database right 2021. This data is licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). Price Paid Data contains address data processed against Ordnance Survey’s AddressBase Premium product, which incorporates Royal Mail’s PAF® database (Address Data).

- [Open Postcode Geo](https://www.getthedata.com/open-postcode-geo)
    - Open Postcode Geo is derived from the ONS Postcode Directory which is licenced under the Open Government Licence and the Ordnance Survey OpenData Licence. See the [ONS Licences](https://www.ons.gov.uk/methodology/geography/licences) page for more information.

- [OpenStreetMap](https://openstreetmap.org/copyright)
    - All the OSM data is available under the Open Database License. The above link has information about OpenStreetMap’s data sources as well as the ODbL.


# About Fynesse Template:

## Fynesse Template

One challenge for data science and data science processes is that they do not always accommodate the real-time and evolving nature of data science advice as required, for example in pandemic response or in managing an international supply chain. The Fynesse paradigm is inspired by experience in operational data science both in the Amazon supply chain and in the UK Covid-19 pandemic response.

The Fynesse paradigm considers three aspects to data analysis, Access, Assess, Address. 

## Access

Gaining access to the data, including overcoming availability challenges (data is distributed across architectures, called from an obscure API, written in log books) as well as legal rights (for example intellectual property rights) and individual privacy rights (such as those provided by the GDPR).

It seems a great challenge to automate all the different aspects of the process of data access, but this challenge is underway already through the process of what is commonly called *digital transformation*. The process of digital transformation takes data away from physical log books and into digital devices. But that transformation process itself comes with challenges. 

Legal complications around data are still a major barrier though. In the EU and the US database schema and indices are subject to copyright law. Companies making data available often require license fees. As many data sources are combined, the composite effect of the different license agreements often makes the legal challenges insurmountable. This was a common challenge in the pandemic, where academics who were capable of dealing with complex data predictions were excluded from data access due to challenges around licensing. A nice counter example was the work led by Nuria Oliver in Spain who after a call to arms in a national newspaper  was able to bring the ecosystem together around mobility data.

However, even when organisation is fully digital, and license issues are overcome, there are issues around how the data is managed stored, accessed. The discoverability of the data and the recording of its provenance are too often neglected in the process of digtial transformation. Further, once an organisation has gone through digital transformation, they begin making predictions around the data. These predictions are data themselves, and their presence in the data ecosystem needs recording. Automating this portion requires structured thinking around our data ecosystems.

## Assess

Understanding what is in the data. Is it what it's purported to be, how are missing values encoded, what are the outliers, what does each variable represent and how is it encoded.

Data that is accessible can be imported (via APIs or database calls or reading a CSV) into the machine and work can be done understanding the nature of the data. The important thing to say about the assess aspect is that it only includes things you can do *without* the question in mind. This runs counter to many ideas about how we do data analytics. The history of statistics was that we think of the question *before* we collect data. But that was because data was expensive, and it needed to be excplicitly collected. The same mantra is true today of *surveillance data*. But the new challenge is around *happenstance data*, data that is cheaply available but may be of poor quality. The nature of the data needs to be understood before its integrated into analysis. Unfortunately, because the work is conflated with other aspects, decisions are sometimes made during assessment (for example approaches to imputing missing values) which may be useful in one context, but are useless in others. So the aim in *assess* is to only do work that is repeatable, and make that work available to others who may also want to use the data.

## Address

The final aspect of the process is to *address* the question. We'll spend the least time on this aspect here, because it's the one that is most widely formally taught and the one that most researchers are familiar with. In statistics, this might involve some confirmatory data analysis. In machine learning it may involve designing a predictive model. In many domains it will involve figuring out how best to visualise the data to present it to those who need to make the decisions. That could involve a dashboard, a plot or even summarisation in an Excel spreadsheet.
