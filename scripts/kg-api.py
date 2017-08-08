#!/usr/bin/env python
"""
Query the Knowledge Graph API https://developers.google.com/knowledge-graph/

"""

import argparse
import datetime
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
import json
import urllib


def main(query):
    api_key = open('api_key.txt').read()
    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'

    params = {
        'query': query,
        'limit': 10,
        'indent': True,
        'key': api_key,
        }

    url = service_url + '?' + urllib.parse.urlencode(params)
    print("Requesting knowledge graph content using " + url)
    response = json.loads(requests.get(url, auth=HTTPDigestAuth('user', 'pass'), headers= {'User-Agent':"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36"}).text)
    
    #NOTE previous approach, using urllib:
    #print("\nRunning json.loads on...\n\n" + str(urllib.request.urlopen(url).read()))
    #response = json.loads(urllib.request.urlopen(url).read().decode("utf-8"))

    # Parsing the response # TODO: Log all responses
    print('Displaying results...' + ' (limit: ' + str(params['limit']) + ')\n')
    #print(response)
    
    for element in response['itemListElement']:
        try:
            types = str(", ".join([n for n in element['result']['@type']]))
        except KeyError:
            types = "N/A"

        try:
            desc = str(element['result']['description'])
        except KeyError:
            desc = "N/A"

        try:
            detail_desc = str(element['result']['detailedDescription']['articleBody'])[0:100] + '...'
        except KeyError:
            detail_desc = "N/A"

        try:
            mid = str(element['result']['@id'])
        except KeyError:
            mid = "N/A"

        try:
            url = str(element['result']['url'])
        except KeyError:
            url = "N/A"

        try:
            score = str(element['resultScore'])
        except KeyError:
            score = "N/A"

        print(element['result']['name'] \
                + '\n' + ' - entity_types: ' + types \
                + '\n' + ' - description: ' + desc \
                + '\n' + ' - detailed_description: ' + detail_desc \
                + '\n' + ' - identifier: ' + mid \
                + '\n' + ' - url: ' + url \
                + '\n' + ' - resultScore: ' + score \
                + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('query', help='The search term to query')
    args = parser.parse_args()
    main(args.query)


"""
Sample result: https://kgsearch.googleapis.com/v1/entities:search?query=taylor+swift&key=[]&limit=1&indent=True

{
  "@context": {
    "@vocab": "http://schema.org/",
    "goog": "http://schema.googleapis.com/",
    "EntitySearchResult": "goog:EntitySearchResult",
    "detailedDescription": "goog:detailedDescription",
    "resultScore": "goog:resultScore",
    "kg": "http://g.co/kg"
  },
  "@type": "ItemList",
  "itemListElement": [
    {
      "@type": "EntitySearchResult",
      "result": {
        "@id": "kg:/m/0dl567",
        "name": "Taylor Swift",
        "@type": [
          "Thing",
          "Person"
        ],
        "description": "Singer-songwriter",
        "image": {
          "contentUrl": "http://t1.gstatic.com/images?q=tbn:ANd9GcQmVDAhjhWnN2OWys2ZMO3PGAhupp5tN2LwF_BJmiHgi19hf8Ku",
          "url": "https://en.wikipedia.org/wiki/Taylor_Swift",
          "license": "http://creativecommons.org/licenses/by-sa/2.0"
        },
        "detailedDescription": {
          "articleBody": "Taylor Alison Swift is an American singer-songwriter.
          Raised in Wyomissing, Pennsylvania, she moved to Nashville, Tennessee, at the age of 14
          to pursue a career in country music. ",
          "url": "http://en.wikipedia.org/wiki/Taylor_Swift",
          "license": "https://en.wikipedia.org/wiki/Wikipedia:Text_of_Creative_Commons_Attribution-ShareAlike_3.0_Unported_License"
        },
        "url": "http://www.taylorswift.com/"
      },
      "resultScore": 884.364868
    }
  ]
}
"""

