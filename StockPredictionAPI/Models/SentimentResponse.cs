using Newtonsoft.Json;
using System.Collections.Generic;

namespace StockPredictionAPI.Models
{
    public class SentimentResponse
    {
        [JsonProperty("items")]
        public List<SentimentData> Items { get; set; }
    }
}