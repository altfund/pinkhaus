# Dashboard Filter Logic Explanation

## Current Behavior (AND Logic)

The dashboard uses **AND logic** between different filter types:
- **Sport AND Nation AND League**

This means a market must match ALL specified filters to be displayed.

### Example:
With filters:
- `ALLOWED_SPORTS="Soccer"`  
- `ALLOWED_NATIONS="England,International"`
- `ALLOWED_LEAGUES=""` (empty = all leagues)

A market is shown if:
- It's Soccer AND
- It's from England OR International nations

## Within Each Filter (OR Logic)

Multiple values within the same filter use **OR logic**:
- `ALLOWED_NATIONS="England,Spain,Italy"` means England OR Spain OR Italy
- `ALLOWED_SPORTS="Soccer,Basketball"` means Soccer OR Basketball

## Common Filter Combinations

### 1. English Soccer Only
```bash
export ALLOWED_SPORTS="Soccer"
export ALLOWED_NATIONS="England"
export ALLOWED_LEAGUES=""
```
Shows: Only soccer matches from England

### 2. Premier League Only
```bash
export ALLOWED_SPORTS="Soccer"
export ALLOWED_NATIONS="England"
export ALLOWED_LEAGUES="Premier League"
```
Shows: Only Premier League matches

### 3. Top European Soccer
```bash
export ALLOWED_SPORTS="Soccer"
export ALLOWED_NATIONS="England,Spain,Italy,Germany,France"
export ALLOWED_LEAGUES=""
```
Shows: Soccer from any of the top 5 European nations

### 4. Champions League & Europa League
```bash
export ALLOWED_SPORTS="Soccer"
export ALLOWED_NATIONS="Europe"
export ALLOWED_LEAGUES="UEFA Champions League,Europa League"
```
Shows: Only European continental competitions

### 5. All International Soccer
```bash
export ALLOWED_SPORTS="Soccer"
export ALLOWED_NATIONS="International,Europe"
export ALLOWED_LEAGUES=""
```
Shows: International matches and European competitions

## Why You Saw Non-Soccer

The MLB, Esports, and MMA matches appeared because they were **incorrectly classified as Soccer** in the database. We've now fixed 520+ misclassified markets:
- 12 MLB markets → Baseball
- 51 Esports markets → Esports  
- 446 American Football markets
- 6 College sports → Basketball/Football
- 5 Cricket markets

Now the Soccer filter will only show actual soccer/football matches!