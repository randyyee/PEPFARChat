import requests
from bs4 import BeautifulSoup

COP_url = 'https://www.state.gov/country-operational-plans/'
CongressCurrentReport_url = 'https://www.state.gov/annual-reports-to-congress-on-the-presidents-emergency-plan-for-aids-relief/'
CongressReport_url = 'https://www.state.gov/pepfar-reports-to-congress-archive/'
MERGuides_url = 'https://help.datim.org/hc/en-us/articles/360000084446-MER-Indicator-Reference-Guides'

response = requests.get(MERGuides_url)

soup = BeautifulSoup(response.text, 'html.parser')

links = soup.find_all('a')

i = 0

for link in links:
    if '.pdf' in link.get('href', []):
        i += 1
        print('Downloading file: ', i)

        response = requests.get(link.get('href'))

        last_word = link.get('href').split('/')[-1].split('.')[0]

        pdf = open('PDFs/' + last_word + '.pdf', 'wb')
        pdf.write(response.content)
        pdf.close()
        print('File ', i, 'downloaded')

print('All PDFs downloaded')
