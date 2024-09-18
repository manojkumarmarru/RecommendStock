using Newtonsoft.Json;
using System;

namespace StockPredictionAPI.Models
{
    public class SentimentData
    {
        [JsonProperty("date")]
        public DateTime Date { get; set; }

        [JsonProperty("sentimentScore")]
        public float SentimentScore { get; set; }
        public float Sentiment { get; set; } // Add this property
    }
}