#!/usr/bin/env python3
"""
Infer governing bodies based on nation and sport type
"""

def get_governing_body_mappings():
    """
    Map nations/organizations to their governing bodies
    """
    return {
        # FIFA Member Associations
        'England': 'FA (The Football Association)',
        'Spain': 'RFEF (Royal Spanish Football Federation)',
        'Italy': 'FIGC (Italian Football Federation)',
        'Germany': 'DFB (German Football Association)',
        'France': 'FFF (French Football Federation)',
        'Portugal': 'FPF (Portuguese Football Federation)',
        'Netherlands': 'KNVB (Royal Dutch Football Association)',
        'Belgium': 'RBFA (Royal Belgian Football Association)',
        'Scotland': 'SFA (Scottish Football Association)',
        'Wales': 'FAW (Football Association of Wales)',
        'Ireland': 'FAI (Football Association of Ireland)',
        'Northern Ireland': 'IFA (Irish Football Association)',
        
        # South American
        'Brazil': 'CBF (Brazilian Football Confederation)',
        'Argentina': 'AFA (Argentine Football Association)',
        'Uruguay': 'AUF (Uruguayan Football Association)',
        'Chile': 'ANFP (National Professional Football Association)',
        'Colombia': 'FCF (Colombian Football Federation)',
        'Peru': 'FPF (Peruvian Football Federation)',
        'Ecuador': 'FEF (Ecuadorian Football Federation)',
        'Paraguay': 'APF (Paraguayan Football Association)',
        'Venezuela': 'FVF (Venezuelan Football Federation)',
        'Bolivia': 'FBF (Bolivian Football Federation)',
        
        # North & Central America
        'USA': 'USSF (United States Soccer Federation)',
        'Mexico': 'FMF (Mexican Football Federation)',
        'Canada': 'CSA (Canadian Soccer Association)',
        'Costa Rica': 'FEDEFUTBOL (Costa Rican Football Federation)',
        
        # Asia
        'Japan': 'JFA (Japan Football Association)',
        'South Korea': 'KFA (Korea Football Association)',
        'Korea': 'KFA (Korea Football Association)',
        'China': 'CFA (Chinese Football Association)',
        'Saudi Arabia': 'SAFF (Saudi Arabian Football Federation)',
        'Iran': 'FFIRI (Football Federation Islamic Republic of Iran)',
        'Australia': 'FFA (Football Federation Australia)',
        'Qatar': 'QFA (Qatar Football Association)',
        'UAE': 'UAEFA (UAE Football Association)',
        'Thailand': 'FAT (Football Association of Thailand)',
        'India': 'AIFF (All India Football Federation)',
        'Indonesia': 'PSSI (Football Association of Indonesia)',
        'Malaysia': 'FAM (Football Association of Malaysia)',
        'Vietnam': 'VFF (Vietnam Football Federation)',
        'Philippines': 'PFF (Philippine Football Federation)',
        'Uzbekistan': 'UFA (Uzbekistan Football Association)',
        
        # Africa
        'Egypt': 'EFA (Egyptian Football Association)',
        'South Africa': 'SAFA (South African Football Association)',
        
        # Europe (Others)
        'Russia': 'RFS (Russian Football Union)',
        'Ukraine': 'UAF (Ukrainian Association of Football)',
        'Turkey': 'TFF (Turkish Football Federation)',
        'Poland': 'PZPN (Polish Football Association)',
        'Romania': 'FRF (Romanian Football Federation)',
        'Czech Republic': 'FACR (Football Association of Czech Republic)',
        'Greece': 'EPO (Hellenic Football Federation)',
        'Austria': 'ÖFB (Austrian Football Association)',
        'Switzerland': 'SFV (Swiss Football Association)',
        'Sweden': 'SvFF (Swedish Football Association)',
        'Norway': 'NFF (Norwegian Football Association)',
        'Denmark': 'DBU (Danish Football Union)',
        'Finland': 'SPL (Football Association of Finland)',
        'Iceland': 'KSI (Football Association of Iceland)',
        'Hungary': 'MLSZ (Hungarian Football Federation)',
        'Croatia': 'HNS (Croatian Football Federation)',
        'Serbia': 'FSS (Football Association of Serbia)',
        'Bulgaria': 'BFU (Bulgarian Football Union)',
        'Slovakia': 'SFZ (Slovak Football Association)',
        'Slovenia': 'NZS (Football Association of Slovenia)',
        'Belarus': 'BFF (Belarus Football Federation)',
        'Estonia': 'EJL (Estonian Football Association)',
        'Latvia': 'LFF (Latvian Football Federation)',
        'Lithuania': 'LFF (Lithuanian Football Federation)',
        'Kazakhstan': 'KFF (Kazakhstan Football Federation)',
        'Israel': 'IFA (Israel Football Association)',
        'Georgia': 'GFF (Georgian Football Federation)',
        'North Macedonia': 'FFM (Football Federation of Macedonia)',
        
        # Continental/International Organizations
        'UEFA': 'UEFA (Union of European Football Associations)',
        'FIFA': 'FIFA (Fédération Internationale de Football Association)',
        'CONMEBOL': 'CONMEBOL (South American Football Confederation)',
        'CONCACAF': 'CONCACAF (Confederation of North, Central America and Caribbean)',
        'AFC': 'AFC (Asian Football Confederation)',
        'CAF': 'CAF (Confederation of African Football)',
        'OFC': 'OFC (Oceania Football Confederation)',
        'COSAFA': 'COSAFA (Council of Southern Africa Football Associations)',
        
        # Basketball
        'FIBA': 'FIBA (International Basketball Federation)',
        
        # Volleyball
        'FIVB': 'FIVB (International Volleyball Federation)',
        'CEV': 'CEV (European Volleyball Confederation)',
        
        # Handball
        'IHF': 'IHF (International Handball Federation)',
        'EHF': 'EHF (European Handball Federation)',
        
        # Ice Hockey
        'IIHF': 'IIHF (International Ice Hockey Federation)',
        
        # Swimming
        'FINA': 'FINA (International Swimming Federation)',
        
        # Cricket
        'ICC': 'ICC (International Cricket Council)',
        
        # Darts
        'PDC': 'PDC (Professional Darts Corporation)',
        
        # Generic/Default mappings
        'International': 'International Federation',
        'Europe': 'European Federation',
        'North America': 'North American Federation',
        'South America': 'South American Federation',
        'Asia': 'Asian Federation',
        'Africa': 'African Federation',
        'Oceania': 'Oceania Federation',
        'Global': 'Global Federation',
        'Unknown': 'Unknown Federation'
    }

def infer_governing_body(nation, sport=None, league=None):
    """
    Intelligently infer governing body from nation, sport, and league
    """
    mappings = get_governing_body_mappings()
    
    # Direct mapping
    if nation in mappings:
        return mappings[nation]
    
    # Sport-specific overrides
    if sport:
        sport_upper = sport.upper()
        
        # US Sports have their own governing bodies
        if nation == 'USA':
            if 'BASKETBALL' in sport_upper or 'NBA' in str(league).upper():
                return 'NBA'
            elif 'FOOTBALL' in sport_upper or 'NFL' in str(league).upper():
                return 'NFL'
            elif 'BASEBALL' in sport_upper or 'MLB' in str(league).upper():
                return 'MLB'
            elif 'HOCKEY' in sport_upper or 'NHL' in str(league).upper():
                return 'NHL'
            elif 'SOCCER' in sport_upper or 'MLS' in str(league).upper():
                return 'USSF (United States Soccer Federation)'
        
        # Basketball federations
        if 'BASKETBALL' in sport_upper:
            if nation in ['Spain', 'Italy', 'Germany', 'France', 'Greece', 'Turkey']:
                return f'{nation} Basketball Federation'
        
        # Esports
        if 'ESPORTS' in sport_upper:
            return 'Various Esports Organizations'
        
        # Tennis
        if 'TENNIS' in sport_upper:
            return 'ITF/ATP/WTA'
        
        # Golf
        if 'GOLF' in sport_upper:
            return 'PGA/European Tour'
        
        # MMA/Fighting
        if 'MMA' in sport_upper or 'UFC' in str(league).upper():
            return 'UFC/MMA Organizations'
    
    # Default by continent/region patterns
    if nation:
        nation_upper = nation.upper()
        if nation_upper in ['FIBA', 'FIVB', 'CEV', 'IHF', 'EHF', 'IIHF', 'FINA', 'ICC', 'PDC']:
            return f'{nation} (Organization)'
        
    # Ultimate fallback
    return f'{nation or "Unknown"} Sports Federation'

if __name__ == "__main__":
    # Test the governing body inference
    print("=== Governing Body Inference Tests ===\n")
    
    test_cases = [
        ('England', 'Soccer', 'Premier League'),
        ('USA', 'Basketball', 'NBA'),
        ('USA', 'Soccer', 'MLS'),
        ('Japan', 'Baseball', 'NPB'),
        ('FIBA', 'Basketball', 'FIBA World Cup'),
        ('UEFA', 'Soccer', 'Champions League'),
        ('Unknown', None, None),
    ]
    
    for nation, sport, league in test_cases:
        gb = infer_governing_body(nation, sport, league)
        print(f"{nation} + {sport} + {league} → {gb}")
    
    print("\n✅ Governing body inference is ready!")