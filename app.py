import streamlit as st
import pandas as pd
import random
import difflib
import requests
from datetime import datetime
from timezonefinder import TimezoneFinder
import pytz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

def fetch_place_images(place_name):
    PIXABAY_API_KEY = "50959863-a3fc0be1d932d4de9f9fb802b"  # Use your key
    url = "https://pixabay.com/api/"
    params = {
        "key": PIXABAY_API_KEY,
        "q": place_name,
        "image_type": "photo",
        "per_page": 3
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return [hit["webformatURL"] for hit in data["hits"]]
    return []

st.set_page_config(page_title="🌏 Asia Travel Matcher", layout="centered")

st.title("🌏 Welcome to the Asia Travel Matcher!")
method = st.radio("Choose how you'd like to get travel suggestions:",
                  ["🧠 Personality Test", "💸 Budget + Theme Preference"])

if "method_selected" not in st.session_state:
    st.session_state.method_selected = False

# ---------- Handle query parameters for country navigation ----------
query_params = st.query_params

if "country" in query_params and not st.session_state.get("started", False):
    st.session_state["country_page"] = query_params["country"]
elif "country_page" not in st.session_state:
    st.session_state["country_page"] = None

if "country" in query_params and st.session_state.get("started", False) and st.session_state.get("country_page") is None:
    query_params.clear()
    st.rerun()

# ---------- Session State Flags ----------
if "started" not in st.session_state:
    st.session_state.started = False
if "use_prompt" not in st.session_state:
    st.session_state.use_prompt = False
if "favourites" not in st.session_state:
    st.session_state.favourites = []
if "question_set_index" not in st.session_state:
    st.session_state["question_set_index"] = random.randint(0, 2)

# ---------- Load dataset ----------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_travel.csv")
    df.fillna("", inplace=True)

    # Ensure required columns exist
    required_cols = ['COUNTRY', 'CITY', 'THEME', 'HIGHLIGHTS', 'PRECAUTION', 'AVG_COST_PER_DAY', 'FLIGHT_COST']
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    # Create necessary derived columns
    df['TOTAL_COST'] = df['AVG_COST_PER_DAY'] + df['FLIGHT_COST']
    df['combined_features'] = (
        df['COUNTRY'].astype(str) + " " +
        df['CITY'].astype(str) + " " +
        df['THEME'].astype(str) + " " +
        df['HIGHLIGHTS'].astype(str) + " " +
        df['PRECAUTION'].astype(str)
    )

    return df


df = load_data()


# ---------- Helper functions for weather ----------
def get_coordinates_osm(location):
    url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return lat, lon
    return None, None

def get_local_time(lat, lon, utc_time_str):
    utc_time = datetime.strptime(utc_time_str, "%Y-%m-%dT%H:%M:%SZ")
    utc_time = utc_time.replace(tzinfo=pytz.utc)

    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=lat, lng=lon)

    if timezone_str:
        local_time = utc_time.astimezone(pytz.timezone(timezone_str))
        return local_time.strftime("%Y-%m-%d %H:%M:%S"), timezone_str
    else:
        return utc_time_str, "UTC"

def get_metno_weather(lat, lon):
    headers = {
        "User-Agent": "weather-checker/1.0 contact@example.com"
    }
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        timeslot = data["properties"]["timeseries"][0]
        timestamp_utc = timeslot["time"]

        details = timeslot["data"]["instant"]["details"]
        temperature = details.get("air_temperature", "N/A")
        humidity = details.get("relative_humidity", "N/A")
        wind_speed = details.get("wind_speed", "N/A")

        condition = "N/A"
        next_data = timeslot["data"].get("next_1_hours") or timeslot["data"].get("next_6_hours")
        if next_data and "summary" in next_data:
            condition = next_data["summary"].get("symbol_code", "N/A").replace("_", " ").capitalize()

        local_time, timezone = get_local_time(lat, lon, timestamp_utc)

        return {
            "local_time": local_time,
            "timezone": timezone,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "condition": condition
        }
    else:
        return None

# ---------- Sidebar Info ----------
if st.session_state.get("started", False):
    st.sidebar.title("🎒 Why This App Exists")
    st.sidebar.info("Lost in travel chaos? This app helps you find your dream destination in Asia — no boring brochures, just good vibes and spicy adventures. 🌶✈🌏")

    st.sidebar.markdown("### 🔟 Top 10 places to visit in Asia before you bye bye ☠")

    top_destinations = [
        ("🇲🇾 Kuala Lumpur", "Malaysia", "🍛"),
        ("🇹🇭 Bangkok", "Thailand", "🛺"),
        ("🇻🇳 Hanoi", "Vietnam", "🛵"),
        ("🇰🇷 Seoul", "South Korea", "🎧"),
        ("🇯🇵 Tokyo", "Japan", "🍣"),
        ("🇮🇳 Jaipur", "India", "🥵"),
        ("🇮🇩 Bali", "Indonesia", "🏄‍♂"),
        ("🇨🇳 Beijing", "China", "🐉"),
        ("🇵🇭 Palawan", "Philippines", "🏝"),
        ("🇸🇬 Singapore", "Singapore", "💸")
    ]

    for i, (city_flag, country, emoji) in enumerate(top_destinations, 1):
        st.sidebar.markdown(
            f'{i}. <a href="?country={country}" target="_self">{city_flag}, {country} {emoji}</a>',
            unsafe_allow_html=True
        )

    with st.sidebar.expander("💖 Bookmarks"):
        if st.session_state.favourites:
            for fav in st.session_state.favourites:
                st.markdown(f"📍 {fav['CITY']}, {fav['COUNTRY']}")
        else:
            st.info("You haven't added any favourite places yet.")

# ---------- Country Detail Page ----------
if st.session_state["country_page"]:
    country_name = st.session_state["country_page"]
    st.title(f"🌏 Explore {country_name}")

    country_data = df[df["COUNTRY"].str.lower() == country_name.lower()]
    if country_data.empty:
        st.warning("No travel data found for this country.")
    else:
        st.subheader("🏙 Popular Cities to Visit:")
        for _, row in country_data.iterrows():
            st.markdown(f"### 📍 {row['CITY']}")
            st.markdown(f"Highlights: {row['HIGHLIGHTS']}")
            st.markdown(f"Flight Cost (from Malaysia): ${row['FLIGHT_COST']}")
            st.markdown(f"Average Daily Cost: ${row['AVG_COST_PER_DAY']}")
            st.markdown(f"Visa Required?: {row['VISA']}")
            st.warning(f"Precaution: {row['PRECAUTION']}")
            st.markdown("---")

        st.subheader("💖 Do you like any place listed above?")
        liked_place = st.text_input("Type the name of a place you liked most from above:", placeholder="e.g. Palawan, Bali, or your favorite spot")

        if st.button("💌 Save to My Bookmark"):
            if liked_place.strip() == "":
                st.warning("Please type a place first before saving!")
            else:
                city_list = country_data['CITY'].tolist()
                best_match = difflib.get_close_matches(liked_place.strip(), city_list, n=1, cutoff=0.6)
                if best_match:
                    corrected_city = best_match[0]
                    st.success(f"Awesome! We got that you meant {corrected_city} in {country_name}. 🌟")
                    st.session_state.favourites.append({"CITY": corrected_city, "COUNTRY": country_name})
                else:
                    st.error("😕 Couldn't match your input with any place listed. Try checking the spelling!")

    if st.button("🔙 Back to Main Page"):
        st.session_state.country_page = None
        st.session_state.started = True
        st.query_params.clear()
        st.rerun()
    st.stop()

# ---------- Question Bank (randomized sets from Script 2) ----------
question_sets = [
    [  # Set 1
        ("Do you enjoy hiking and exploring nature?", "nature"),
        ("Do you prefer a bustling city over a quiet village?", "urban"),
        ("Are historical landmarks important to you when traveling?", "culture"),
        ("Do you often try street food in new places?", "food"),
        ("Is nightlife a key part of your travel experience?", "party"),
        ("Would you prefer a yoga retreat over a hiking trip?", "relax"),
        ("Do you like water sports or extreme adventures?", "adventure"),
        ("Are you fascinated by temples and ancient ruins?", "culture"),
        ("Would you enjoy a day in a traditional market?", "food"),
        ("Is shopping in modern malls or tech hubs exciting for you?", "urban"),
        ("Do you often go to clubs or festivals?", "party"),
        ("Do you enjoy national parks and scenic views?", "nature"),
        ("Would you book a spa day during your vacation?", "relax"),
        ("Do you enjoy road trips through mountainous regions?", "adventure"),
        ("Is art and museum hopping part of your trip?", "culture"),
        ("Would you rather relax on a beach than explore a museum?", "relax"),
        ("Do you love tasting exotic or spicy dishes?", "food"),
        ("Are you more excited by skyscrapers than rice fields?", "urban"),
        ("Do you enjoy nature documentaries or wildlife tours?", "nature"),
        ("Would you skydive or bungee jump on vacation?", "adventure"),
    ],
    [  # Set 2
        ("Would you spend a night in a treehouse or eco-lodge?", "nature"),
        ("Do bright city lights excite you more than starry skies?", "urban"),
        ("Do you love learning local history when traveling?", "culture"),
        ("Would you try exotic snacks at a market stall?", "food"),
        ("Is attending concerts or nightlife important on your trips?", "party"),
        ("Would you go on a silent retreat or wellness spa?", "relax"),
        ("Do you enjoy rafting, ziplining or ATV rides?", "adventure"),
        ("Do temples and ancient architecture fascinate you?", "culture"),
        ("Do you enjoy local cooking classes or food tours?", "food"),
        ("Would you rather explore skyscrapers than forests?", "urban"),
        ("Are music festivals a must-do for you?", "party"),
        ("Would you hike to a hidden waterfall?", "nature"),
        ("Do you take time for massages or self-care while traveling?", "relax"),
        ("Would you drive through desert roads or rugged terrains?", "adventure"),
        ("Is visiting UNESCO sites part of your dream trip?", "culture"),
        ("Would you relax by a lake over a shopping district?", "relax"),
        ("Do you enjoy sampling regional dishes with your hands?", "food"),
        ("Do you prefer urban cafes over countryside tea houses?", "urban"),
        ("Would you love a wildlife safari or jungle trek?", "nature"),
        ("Would you try paragliding, diving or cliff jumping?", "adventure"),
    ],
    [  # Set 3
        ("Would you enjoy staying in a bamboo cabin in the mountains?", "nature"),
        ("Are you drawn to tech-forward cities and smart hotels?", "urban"),
        ("Do you prefer visiting temples over shopping malls?", "culture"),
        ("Do you eat local breakfast dishes instead of hotel buffets?", "food"),
        ("Do you plan trips around events and nightlife?", "party"),
        ("Would you meditate on a mountain top?", "relax"),
        ("Is hiking to remote places part of your adventure?", "adventure"),
        ("Do you enjoy ancient cities with centuries of history?", "culture"),
        ("Do you prioritize trying local drinks and delicacies?", "food"),
        ("Would you choose a metro ride over a countryside train?", "urban"),
        ("Is dancing at beach parties or music nights fun for you?", "party"),
        ("Would you swim in a natural spring or mountain stream?", "nature"),
        ("Do you like to read or write while on vacation?", "relax"),
        ("Would you join mountain trekking or desert rallies?", "adventure"),
        ("Is local tradition more important than modern comfort?", "culture"),
        ("Would you do sunrise yoga on a beach?", "relax"),
        ("Would you try unfamiliar food combinations just to explore?", "food"),
        ("Do you enjoy rooftop views of busy cities?", "urban"),
        ("Would you love seeing animals in their natural habitat?", "nature"),
        ("Would you try hang gliding or cave diving?", "adventure"),
    ]
]
questions = question_sets[st.session_state["question_set_index"]]

category_map = {
    "nature": "Photography",
    "urban": "Modern",
    "culture": "Culture",
    "food": "Culture",
    "party": "Modern",
    "relax": "Adventurous",
    "adventure": "Adventurous"
}

funny_theme_labels = {
    "Photography": "🦜 Nature Nerd",
    "Modern": "🌆 City Hustler",
    "Culture": "🏯 Culture Buff",
    "Adventurous": "⛰ Adrenaline Junkie"
}

# ---------- Intro ----------
if not st.session_state.started:
    st.title("👋 Welcome to the Asia Travel Personality Bot!")
    st.markdown("### Discover your travel style in questions. 🧓")
    if st.button("🚀 Start My Journey"):
        st.session_state.started = True
        st.rerun()
    st.stop()

# ---------- Quiz ----------
if method == "🧠 Personality Test":
    st.title("🌏 Asia Travel Personality Chatbot – Find Your Inner Explorer 🤸‍♂🍜🙺")

    theme_scores = {k: 0 for k in category_map}
    with st.form("quiz_form"):
        for idx, (question, category) in enumerate(questions):
            answer = st.radio(f"{idx + 1}. {question}", ["Yes", "No"], key=f"q{idx}")
            if answer == "Yes":
                theme_scores[category] += 1
        if st.form_submit_button("🎯 Get My Destination"):
            top_category = max(theme_scores, key=theme_scores.get)
            matched_theme = category_map.get(top_category, "Culture")
            st.session_state.matched_destination = df[df["THEME"].str.lower() == matched_theme.lower()].sample(1).iloc[
                0].to_dict()
            st.session_state.theme_scores = theme_scores
            st.session_state.top_category = top_category
            st.rerun()

    # ---------- Display Destination ----------
    if "matched_destination" in st.session_state:
        match = st.session_state.matched_destination
        top_category = st.session_state.top_category
        matched_theme = category_map.get(top_category, "Culture")
        funny_label = funny_theme_labels.get(matched_theme, matched_theme)
        st.markdown(f"### 🧭 You are a :orange[{funny_label}]!")

        with st.expander("📍 Your Perfect Travel Soulmate", expanded=True):

            st.subheader(f"📌 {match['CITY']}, {match['COUNTRY']} – Let's gooo! 💒")
            with st.spinner("🖼 Fetching inspiring images..."):
                images = fetch_place_images(f"{match['CITY']}, {match['COUNTRY']}")
                if images:
                    st.image(images, width=300)
                else:
                    st.warning("Couldn't find images for this location.")

            st.markdown(f"🏜 Why You’ll Love It**: {match['HIGHLIGHTS']} 💖")
            st.markdown(f"✈ Flight Cost: ${match['FLIGHT_COST']}")
            st.markdown(f"🛎 Daily Burn Rate: ${match['AVG_COST_PER_DAY']}")
            st.session_state.trip_days = st.number_input("🗓 How many days?", min_value=1, value=7)
            total_cost = match['FLIGHT_COST'] + match['AVG_COST_PER_DAY'] * st.session_state.trip_days
            st.success(f"💰 Total Cost for {int(st.session_state.trip_days)} days: ${total_cost:.2f}")
            with st.spinner("⛅ Fetching current weather..."):
                lat, lon = get_coordinates_osm(f"{match['CITY']}, {match['COUNTRY']}")
                if lat and lon:
                    weather = get_metno_weather(lat, lon)
                    if weather:
                        st.info(
                            f"🕒 Local Time: {weather['local_time']} ({weather['timezone']})\n\n"
                            f"🌡 Temperature: {weather['temperature']}°C\n\n"
                            f"💧 Humidity: {weather['humidity']}%\n\n"
                            f"🌬 Wind: {weather['wind_speed']} m/s\n\n"
                            f"☁ Condition: {weather['condition']}"
                        )
                    else:
                        st.warning("⚠ Weather data is unavailable.")
                else:
                    st.warning("⚠ Could not find coordinates for this destination.")
        with st.expander("📄 Travel Essentials"):
            if match['VISA'].strip().lower() == "no":
                st.success("🟢 No visa required!")
            else:
                st.warning("🔴 Visa required!")
            st.warning(f"⚠ Caution: {match['PRECAUTION']}")

        if st.button("💖 Add this to My Bookmark"):
            favourite = {"CITY": match["CITY"], "COUNTRY": match["COUNTRY"]}
            if favourite not in st.session_state.favourites:
                st.session_state.favourites.append(favourite)
                st.success(f"✅ Added {match['CITY']} to your favourites!")
            else:
                st.info(f"📍 {match['CITY']} is already in your favourites.")

        if st.button("🔁 Retake Quiz"):
            # Clear quiz answers
            for idx in range(len(questions)):
                st.session_state.pop(f"q{idx}", None)
            # Clear result-related states
            st.session_state.pop("matched_destination", None)
            st.session_state.pop("theme_scores", None)
            st.session_state.pop("top_category", None)
            # Optionally pick a new question set
            st.session_state["question_set_index"] = random.randint(0, 2)


        # --- Immediately present chatbot after destination is shown ---
        if "chat_step" not in st.session_state:
            st.session_state.chat_step = "liking_check"
            st.session_state.chat_history = [("bot", "Is this location to your liking?")]

        # ---------- Chatbot Interaction ----------
        st.markdown("---")
        st.subheader("🤖 Chat with TravelBot")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Show chat history
        for speaker, message in st.session_state.chat_history:
            if speaker == "bot":
                st.markdown(f"*TravelBot:* {message}")
            else:
                st.markdown(f"*You:* {message}")

        # Step: manage input and prevent re-processing
        if "awaiting_input" not in st.session_state:
            st.session_state.awaiting_input = True

        # Input box (only when waiting for user reply)
        if st.session_state.awaiting_input:
            user_input = st.text_input("Type your reply here:", key="chat_input")
            if user_input:
                st.session_state.awaiting_input = False
                st.rerun()

        # Process input after rerun
        elif not st.session_state.awaiting_input and "chat_input" in st.session_state:
            user_msg = st.session_state.chat_input.strip()
            st.session_state.chat_history.append(("user", user_msg))
            user_msg_lower = user_msg.lower()

            # === Step 1: Like the location? ===
            if st.session_state.chat_step == "liking_check":
                if "yes" in user_msg_lower:
                    bot_msg = "That's great!! Would you like to save this location to your 'Bookmark'?"
                    st.session_state.chat_step = "offer_save"
                elif "no" in user_msg_lower:
                    bot_msg = "That's too bad. Would you like to retake a different set of test questions?"
                    st.session_state.chat_step = "offer_retake"
                else:
                    bot_msg = "Please respond to the relevant prompt: 'yes' or 'no'."
                st.session_state.chat_history.append(("bot", bot_msg))

            # === Step 2: Offer to save/bookmark ===
            elif st.session_state.chat_step == "offer_save":
                if "yes" in user_msg_lower:
                    favourite = {"CITY": match["CITY"], "COUNTRY": match["COUNTRY"]}
                    if favourite not in st.session_state.favourites:
                        st.session_state.favourites.append(favourite)
                        bot_msg = f"✅ {match['CITY']} saved to your favourites! You can view our top 10 best places to visit in Asia on the sidebar to the left. Or would you like to explore any country?"
                    else:
                        bot_msg = f"📍 {match['CITY']} is already in your favourites! And don’t forget to check the sidebar for Asia’s top 10 hot spots!"
                    st.session_state.chat_step = "done"
                elif "no" in user_msg_lower:
                    bot_msg = "No worries! If you're curious, check out the sidebar's top 10 Asia gems 🌏"
                    st.session_state.chat_step = "done"
                else:
                    bot_msg = "Please respond to the relevant prompt: 'yes' or 'no'."
                st.session_state.chat_history.append(("bot", bot_msg))

            # === Step 3: Retake test? ===
            elif st.session_state.chat_step == "offer_retake":
                if "yes" in user_msg_lower:
                    for idx in range(len(questions)):
                        st.session_state.pop(f"q{idx}", None)
                    st.session_state.pop("matched_destination", None)
                    st.session_state.pop("theme_scores", None)
                    st.session_state.pop("top_category", None)
                    st.session_state.pop("chat_step", None)
                    st.session_state.pop("chat_history", None)
                    st.session_state["question_set_index"] = random.randint(0, 2)
                    st.rerun()
                elif "no" in user_msg_lower:
                    bot_msg = "Do you have a desired location in mind?"
                    st.session_state.chat_step = "custom_location"
                    st.session_state.chat_history.append(("bot", bot_msg))
                else:
                    st.session_state.chat_history.append(("bot", "Please respond to the relevant prompt: 'yes' or 'no'."))

            # === Step 4: Custom Location Search ===
            elif st.session_state.chat_step == "custom_location":
                user_location = user_msg.strip().title()
                asian_countries = df["COUNTRY"].unique()
                asian_cities = df["CITY"].unique()

                matched_country = difflib.get_close_matches(user_location, asian_countries, n=1, cutoff=0.6)
                matched_city = difflib.get_close_matches(user_location, asian_cities, n=1, cutoff=0.6)

                if matched_city:
                    city_data = df[df['CITY'] == matched_city[0]].iloc[0]
                    info = (
                        f"📍 *{city_data['CITY']}, {city_data['COUNTRY']}*\n\n"
                        f"- Highlights: {city_data['HIGHLIGHTS']}\n"
                        f"- Flight Cost: ${city_data['FLIGHT_COST']}\n"
                        f"- Avg Daily Cost: ${city_data['AVG_COST_PER_DAY']}\n"
                        f"- Visa: {city_data['VISA']}\n"
                        f"- Precaution: {city_data['PRECAUTION']}\n"
                        "---"
                    )
                    st.session_state.chat_history.append(("bot", info))
                    st.session_state.chat_step = "done"

                elif matched_country:
                    country_data = df[df['COUNTRY'] == matched_country[0]]
                    bot_msg = f"Here's what I found for *{matched_country[0]}*:"
                    st.session_state.chat_history.append(("bot", bot_msg))
                    for _, row in country_data.iterrows():
                        info = (
                            f"📍 *{row['CITY']}*\n\n"
                            f"- Highlights: {row['HIGHLIGHTS']}\n"
                            f"- Flight Cost: ${row['FLIGHT_COST']}\n"
                            f"- Avg Daily Cost: ${row['AVG_COST_PER_DAY']}\n"
                            f"- Visa: {row['VISA']}\n"
                            f"- Precaution: {row['PRECAUTION']}\n"
                            "---"
                        )
                        st.session_state.chat_history.append(("bot", info))
                    st.session_state.chat_step = "done"

                else:
                    st.session_state.chat_history.append((
                        "bot",
                        f"Sorry, I couldn't find any information for '{user_location}'. Please make sure it's a country or city in Asia."
                    ))

            # === Step 5: Done State ===
            elif st.session_state.chat_step == "done":
                user_location = user_msg.strip().title()
                asian_countries = df["COUNTRY"].unique()

                matched_country = difflib.get_close_matches(user_location, asian_countries, n=1, cutoff=0.6)

                if matched_country:
                    country_name = matched_country[0]
                    country_data = df[df['COUNTRY'] == country_name]

                    if not country_data.empty:
                        selected_city = country_data.sample(1).iloc[0]
                        info = (
                            f"🌏 Here's a place in *{country_name}* you might love!\n\n"
                            f"📍 *{selected_city['CITY']}*\n"
                            f"- Highlights: {selected_city['HIGHLIGHTS']}\n"
                            f"- Flight Cost: ${selected_city['FLIGHT_COST']}\n"
                            f"- Avg Daily Cost: ${selected_city['AVG_COST_PER_DAY']}\n"
                            f"- Visa: {selected_city['VISA']}\n"
                            f"- Precaution: {selected_city['PRECAUTION']}\n"
                            "---"
                        )
                        st.session_state.chat_history.append(("bot", info))
                    else:
                        st.session_state.chat_history.append(("bot", f"⚠ No cities found in {country_name}."))
                else:
                    st.session_state.chat_history.append((
                        "bot",
                        f"❌ Sorry, I couldn't find any info for '{user_location}'. Try a valid Asian country name."
                    ))

                st.session_state.chat_history.append(("bot", "Let me know if you'd like to explore another country 🌏"))

            # Clear input and allow new input
            st.session_state.chat_input = ""
            st.session_state.awaiting_input = True
            st.rerun()

if method == "💸 Budget + Theme Preference":
    # ---------- TF-IDF + Cost ----------
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

    scaler = MinMaxScaler()
    cost_scaled_all = scaler.fit_transform(df[['TOTAL_COST']])
    cost_weight = 1.5
    cost_scaled_all *= cost_weight

    features_all = hstack([tfidf_matrix, cost_scaled_all])


    # ---------- Country Penalty ----------
    def apply_country_penalty(df, similarities, max_per_country=2, top_n=5):
        selected = []
        used_countries = {}
        for idx in similarities.argsort()[::-1]:
            country = df.iloc[idx]['COUNTRY']
            if used_countries.get(country, 0) < max_per_country:
                selected.append(idx)
                used_countries[country] = used_countries.get(country, 0) + 1
            if len(selected) >= top_n:
                break
        return selected


    # ---------- Recommendation Function ----------
    def recommend_destinations(user_input, top_n=5):
        user_total_budget = user_input['budget'] + user_input['flight']
        tolerance = 0.10
        budget_min = user_total_budget * (1 - tolerance)
        budget_max = user_total_budget * (1 + tolerance)

        filtered_df = df[(df['TOTAL_COST'] >= budget_min) & (df['TOTAL_COST'] <= budget_max)]

        if filtered_df.empty:
            return pd.DataFrame(columns=['COUNTRY', 'CITY', 'THEME', 'AVG_COST_PER_DAY', 'FLIGHT_COST', 'HIGHLIGHTS'])

        tfidf_filtered = vectorizer.transform(filtered_df['combined_features'])
        cost_scaled_filtered = scaler.transform(filtered_df[['TOTAL_COST']]) * cost_weight
        features_filtered = hstack([tfidf_filtered, cost_scaled_filtered])

        user_query = user_input['theme'] + " " + user_input['precaution']
        user_tfidf = vectorizer.transform([user_query])
        user_cost_scaled = scaler.transform([[user_total_budget]]) * cost_weight
        user_feature = hstack([user_tfidf, user_cost_scaled])

        similarities = cosine_similarity(user_feature, features_filtered).flatten()
        selected_indices = apply_country_penalty(filtered_df.reset_index(drop=True), similarities, max_per_country=2,
                                                 top_n=top_n)

        return filtered_df.iloc[selected_indices][
            ['COUNTRY', 'CITY', 'THEME', 'AVG_COST_PER_DAY', 'FLIGHT_COST', 'HIGHLIGHTS']]


    # ---------- Streamlit UI ----------
    st.title("🌍 Smart Travel Recommender with Budget & Diversity")

    st.subheader("✈ Travel Preferences")
    theme = st.text_input("Preferred Themes (e.g., culture foodie temple)", "culture foodie temple")
    precaution = st.text_input("Precautions (e.g., cold, safe)", "cold")
    budget = st.slider("Daily Budget (USD)", 10, 500, 90)
    flight = st.slider("Flight Budget (USD)", 50, 2000, 700)
    top_n = st.slider("Number of Recommendations", 3, 5, 5)

    user_input = {
        "theme": theme,
        "precaution": precaution,
        "budget": budget,
        "flight": flight
    }

    # Run recommendation
    results = recommend_destinations(user_input, top_n=top_n)

    # Show results
    if results.empty:
        st.warning("⚠ No destinations found within your budget range. Try increasing the budget or relaxing filters.")
    else:
        st.success(f"🎯 Top {top_n} Travel Recommendations for Your Preferences:")
        st.dataframe(results.reset_index(drop=True))

# ---------- Footer ----------
st.markdown("---")
st.caption("🚀 Built by AI, powered by wanderlust. OpenAI-enhanced travel fun mode active ✨")