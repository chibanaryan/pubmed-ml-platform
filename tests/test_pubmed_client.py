"""Tests for the PubMed E-utilities API client."""

from datetime import date
from unittest.mock import MagicMock, patch
from xml.etree import ElementTree

import pytest

from src.ingestion.pubmed_client import PubMedClient, CATEGORIES


SAMPLE_SEARCH_RESPONSE = {
    "esearchresult": {
        "count": "42",
        "idlist": ["12345", "67890"],
    }
}

SAMPLE_FETCH_XML = """<?xml version="1.0" ?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>Creatine and muscle recovery in athletes</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Creatine is widely used.</AbstractText>
          <AbstractText Label="RESULTS">Significant improvement observed.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Smith</LastName><ForeName>John</ForeName></Author>
          <Author><LastName>Doe</LastName><ForeName>Jane</ForeName></Author>
        </AuthorList>
        <Journal><Title>Journal of Sports Medicine</Title></Journal>
        <ArticleIdList>
          <ArticleId IdType="doi">10.1234/test</ArticleId>
        </ArticleIdList>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Creatine</DescriptorName></MeshHeading>
        <MeshHeading><DescriptorName>Muscle, Skeletal</DescriptorName></MeshHeading>
      </MeshHeadingList>
      <KeywordList>
        <Keyword>supplementation</Keyword>
      </KeywordList>
    </MedlineCitation>
    <PubmedData>
      <History>
        <PubMedPubDate PubStatus="pubmed">
          <Year>2024</Year><Month>03</Month><Day>15</Day>
        </PubMedPubDate>
      </History>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1234/test</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>"""


class TestPubMedClient:
    def test_search_returns_pmids_and_count(self):
        client = PubMedClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_SEARCH_RESPONSE

        with patch.object(client, "_get", return_value=mock_resp):
            pmids, total = client.search("test query")

        assert pmids == [12345, 67890]
        assert total == 42

    def test_search_with_date_filters(self):
        client = PubMedClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_SEARCH_RESPONSE

        with patch.object(client, "_get", return_value=mock_resp) as mock_get:
            client.search(
                "test",
                min_date=date(2024, 1, 1),
                max_date=date(2024, 12, 31),
            )

        params = mock_get.call_args[1]["params"] if mock_get.call_args[1] else mock_get.call_args[0][1]
        assert params["mindate"] == "2024/01/01"
        assert params["maxdate"] == "2024/12/31"

    def test_fetch_articles_parses_xml(self):
        client = PubMedClient()
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_FETCH_XML

        with patch.object(client, "_get", return_value=mock_resp):
            articles = client.fetch_articles([12345])

        assert len(articles) == 1
        a = articles[0]
        assert a.pmid == 12345
        assert a.title == "Creatine and muscle recovery in athletes"
        assert "BACKGROUND: Creatine is widely used." in a.abstract
        assert "RESULTS: Significant improvement observed." in a.abstract
        assert a.authors == ["Smith, John", "Doe, Jane"]
        assert a.journal == "Journal of Sports Medicine"
        assert a.mesh_terms == ["Creatine", "Muscle, Skeletal"]
        assert a.keywords == ["supplementation"]
        assert a.doi == "10.1234/test"

    def test_fetch_articles_handles_missing_abstract(self):
        xml = """<?xml version="1.0" ?>
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>99999</PMID>
              <Article>
                <ArticleTitle>No abstract here</ArticleTitle>
              </Article>
            </MedlineCitation>
          </PubmedArticle>
        </PubmedArticleSet>"""

        client = PubMedClient()
        mock_resp = MagicMock()
        mock_resp.text = xml

        with patch.object(client, "_get", return_value=mock_resp):
            articles = client.fetch_articles([99999])

        assert len(articles) == 1
        assert articles[0].abstract is None

    def test_throttle_respects_rate_limit(self):
        client = PubMedClient(requests_per_second=10.0)
        assert client.min_interval == pytest.approx(0.1, abs=0.01)

    def test_api_key_increases_rate(self):
        client = PubMedClient(api_key="test_key")
        assert client.min_interval == pytest.approx(0.1, abs=0.01)

    def test_categories_are_defined(self):
        assert "nutrition" in CATEGORIES
        assert "exercise" in CATEGORIES
        assert "psychology" in CATEGORIES
        assert "habits" in CATEGORIES
        assert "ethics" in CATEGORIES

    def test_parse_pub_date_text_month(self):
        client = PubMedClient()
        xml = """<PubmedArticle>
            <MedlineCitation>
                <PMID>1</PMID>
                <Article>
                    <ArticleTitle>Test</ArticleTitle>
                    <Journal>
                        <Title>Test</Title>
                        <JournalIssue>
                            <PubDate>
                                <Year>2024</Year><Month>Mar</Month><Day>01</Day>
                            </PubDate>
                        </JournalIssue>
                    </Journal>
                </Article>
            </MedlineCitation>
        </PubmedArticle>"""
        elem = ElementTree.fromstring(xml)
        article = client._parse_article(elem)
        assert article.pub_date == date(2024, 3, 1)
