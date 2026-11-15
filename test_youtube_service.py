"""Offline unit tests: no external API calls or secrets required."""
import unittest
from unittest.mock import patch
from youtube_service import parse_video_id, search_videos, transcript_analysis

class YouTubeServiceTests(unittest.TestCase):
    def test_video_url_parsing(self):
        vid = "dQw4w9WgXcQ"
        for value in [vid, "https://www.youtube.com/watch?v="+vid, "https://youtu.be/"+vid,
                      "https://www.youtube.com/shorts/"+vid]:
            self.assertEqual(parse_video_id(value), vid)
    def test_rejects_untrusted_hosts(self):
        self.assertIsNone(parse_video_id("https://youtube.com.attacker.example/watch?v=dQw4w9WgXcQ"))
        self.assertIsNone(parse_video_id("https://example.org/watch?v=dQw4w9WgXcQ"))
    def test_query_validation(self):
        with self.assertRaises(ValueError):
            search_videos("x")
    def test_transcript_validation(self):
        with self.assertRaises(ValueError):
            transcript_analysis("invalid", "x"*100)
        with self.assertRaises(ValueError):
            transcript_analysis("dQw4w9WgXcQ", "short")
        with self.assertRaises(ValueError):
            transcript_analysis("dQw4w9WgXcQ", "x"*100, language="invalid")
    @patch("youtube_service.youtube_api")
    def test_search_maps_official_metadata(self, youtube_api):
        youtube_api.return_value = {"items":[{"id":"dQw4w9WgXcQ","snippet":{"title":"Demo","channelTitle":"Test","thumbnails":{"medium":{"url":"https://i.ytimg.com/example.jpg"}}},"contentDetails":{"duration":"PT1M"}}]}
        result = search_videos("https://youtu.be/dQw4w9WgXcQ")
        self.assertEqual(result[0]["id"], "dQw4w9WgXcQ")
        self.assertEqual(result[0]["channel"], "Test")

if __name__ == "__main__":
    unittest.main()
