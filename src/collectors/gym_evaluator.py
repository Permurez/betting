import urllib.request
import urllib.parse
import json
import re
import logging
from typing import Dict, Tuple

# Skala Tierów dla klubów (1-5 gwiazdek)
# Oceny oparte na liczbie mistrzów UFC/KSW/Fame i zapleczu trenerskim.
GYM_TIERS = {
    # TIER 5 - Elita Światowa / Elita Polska (Mistrzowie UFC / KSW)
    "american top team": 5,
    "att": 5,
    "city kickboxing": 5,
    "jackson-wink": 5,
    "jackson wink": 5,
    "aka": 5,
    "american kickboxing academy": 5,
    "wca": 5,
    "czerwony smok": 5,
    "ankos": 5,
    "kill cliff": 5,
    "xtreme couture": 5,
    
    # TIER 4 - Bardzo mocne kluby, solidne zaplecze
    "sbg": 4,
    "roufusport": 4,
    "tiger muay thai": 4,
    "tristar": 4,
    "serra-longo": 4,
    "berserkers": 4,
    "octopus": 4,
    "silesian": 4,
    "uniejów": 4,
    
    # TIER 3 - Solidne kluby lokalne
    "team quest": 3,
    "alliance mma": 3,
    "trento": 3,
    "bloodline": 3,
}

class GymEvaluator:
    def __init__(self):
        self._cache = {} # Cache w pamięci by nie odpytywać API w kółko
        
    def evaluate_fighter_gym(self, fighter_name: str) -> Tuple[int, str]:
        """
        Dla danego zawodnika szuka jego klubu (Wikipedia API - darmowe i bez kluczy),
        a następnie ocenia pod kątem bazy eksperckiej (1-5 pkt).
        """
        if fighter_name in self._cache:
            return self._cache[fighter_name]
            
        try:
            # Używamy natywnego urllib dla stabilności (dodano redirects=1 w API Wiki)
            title = urllib.parse.quote(fighter_name.replace(" ", "_"))
            url = f"https://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&titles={title}&redirects=1&format=json&rvslots=main"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 QuantBetBot'})
            
            with urllib.request.urlopen(req, timeout=10.0) as response:
                payload = response.read().decode('utf-8')
                data = json.loads(payload)
            
            pages = data.get("query", {}).get("pages", {})
            if not pages or "-1" in pages:
                self._cache[fighter_name] = (2, "Unknown/Nicely Niche")
                return 2, "Unknown/Nicely Niche"
                
            page = list(pages.values())[0]
            content = page.get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("*", "").lower()
            
            # Ekstrakcja pola "team" lub "gym" z infoboxa, zabezpieczona pod multiline 'ubl' z re.DOTALL
            team_match = re.search(r'\|\s*(?:team|gym)\s*=\s*(.*?)(?=\n\s*\|[a-z_0-9]+\s*=)', content, re.DOTALL)
            
            if team_match:
                extracted_gym = team_match.group(1).strip().lower()
                # Czyszczenie śmieci ze specyficznych znaczników Wikipedii
                extracted_gym = re.sub(r'\{\{(?:ubl|plainlist|unbulleted list)\|?', '', extracted_gym)
                extracted_gym = extracted_gym.replace("[", "").replace("]", "").replace("{", "").replace("}", "")
                
                for gym_name, tier in GYM_TIERS.items():
                    # Używamy word bounds (\b) zeby skróty jak "att" (American Top Team) 
                    # nie łapały słów typu "attack" lub "pattern".
                    if re.search(r'\b' + re.escape(gym_name) + r'\b', extracted_gym):
                        self._cache[fighter_name] = (tier, gym_name)
                        return tier, gym_name.title()
                
                # Znaleziono klub, ale nie ma go w elitarnej topce - Tier 3
                # Czyścimy wynik do logowania z resztek tagów 
                fallback_name = extracted_gym.split('\n')[0][:30].title() + "..."
                self._cache[fighter_name] = (3, fallback_name)
                return 3, fallback_name
            
            self._cache[fighter_name] = (2, "No Team Rec.")
            return 2, "No Team Rec."
            
        except Exception as e:
            logging.warning(f"Błąd wyszukiwania dla {fighter_name}: {e}")
            self._cache[fighter_name] = (2, "Error/NotFound")
            return 2, "Error/NotFound"
            
    def get_camp_investment_estimate(self, fighter_name: str) -> float:
        tier, _ = self.evaluate_fighter_gym(fighter_name)
        base_investment = float(tier) * 2.0 
        return min(10.0, base_investment)

if __name__ == "__main__":
    evaluator = GymEvaluator()
    print("Testowanie zautomatyzowanego narzędzia GymEvaluator (Wikipedia OSINT):")
    for f in ["Jon Jones", "Conor McGregor", "Jan Blachowicz", "Mateusz Gamrot", "Random Fighter 123"]:
        t, g = evaluator.evaluate_fighter_gym(f)
        invest = evaluator.get_camp_investment_estimate(f)
        print(f"Zawodnik: {f:<20} | Klub: {g:<25} | Tier: {t} | Budżet (1-10): {invest}")
