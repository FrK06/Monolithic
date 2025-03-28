import requests
from typing import Dict, List
import os
from urllib.parse import urlparse
from collections import defaultdict

class WebSearcher:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        self.endpoint = "https://www.googleapis.com/customsearch/v1"
        self.default_max_results = 10  # Increase from 5 to ensure we have enough results

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": max_results,
            "lr": "lang_en",      # English results only
            "safe": "active",     # Filter explicit content
            "sort": "date:r:0",   # Sort by date - recent first
            "filter": "1"         # Filter duplicate content
        }

        time_fallback = [
            {"link": "https://www.timeanddate.com/worldclock/", "title": "Time and Date", "snippet": "Current time reference"},
            {"link": "https://www.worldtimebuddy.com/", "title": "World Time Buddy", "snippet": "Global time zones"},
        ]

        try:
            response = requests.get(self.endpoint, params=params)
            response.raise_for_status()
            results = response.json()

            if 'items' in results and results['items']:
                search_results = [
                    {
                        "link": item["link"],
                        "title": item["title"],
                        "snippet": item["snippet"],
                        "provider": "Google",
                        "date": item.get("pagemap", {}).get("metatags", [{}])[0].get("article:published_time", "")
                    }
                    for item in results["items"]
                ]
                
                # If we have more than 5 results, pick the 5 most relevant/diverse
                if len(search_results) > 5:
                    return self._select_diverse_results(search_results, 5)
                return search_results
            else:
                return []  # No results found
        except Exception as e:
            print(f"Google search error: {str(e)}")
            # Fallback to time-related sites if query is time-related
            if any(word in query.lower() for word in ['time', 'date', 'timezone', 'clock']):
                return time_fallback
            else:
                return []
    
    def _select_diverse_results(self, results: List[Dict[str, str]], num_results: int = 5) -> List[Dict[str, str]]:
        """
        Select a diverse set of results by choosing from different domains
        and ensuring a mix of recent and relevant results.
        """
        # Extract domains from links
        for result in results:
            result['domain'] = urlparse(result['link']).netloc
        
        # Group by domain
        domain_groups = defaultdict(list)
        for result in results:
            domain_groups[result['domain']].append(result)
        
        # Select the best result from each domain until we have enough
        selected_results = []
        domains = list(domain_groups.keys())
        
        # First, select one result from each domain
        for domain in domains:
            if len(selected_results) >= num_results:
                break
            # Pick the first (usually most relevant) result from each domain
            selected_results.append(domain_groups[domain][0])
            # Remove this domain from further consideration
            domains.remove(domain)
        
        # If we still need more results, go back and select second results from domains
        domains = list(domain_groups.keys())
        i = 0
        while len(selected_results) < num_results and i < len(domains):
            domain = domains[i]
            if len(domain_groups[domain]) > 1:
                selected_results.append(domain_groups[domain][1])
            i += 1
            
        # Ensure we've got exactly the number requested
        return selected_results[:num_results]
