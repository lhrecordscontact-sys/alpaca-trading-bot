# ============================================================
# NEXT DAILY SCAN TIME
# ============================================================

def get_next_scan_time():

    now = datetime.now(NY)

    target = now.replace(
        hour=SCAN_HOUR_ET,
        minute=SCAN_MINUTE_ET,
        second=0,
        microsecond=0
    )

    # If today's scan time already passed,
    # schedule tomorrow.
    if target <= now:
        target += timedelta(days=1)

    return target


# ============================================================
# AUTOMATIC DAILY SCANNER LOOP
#
# ONE scan EVERY DAY at 4:15 PM ET.
# Includes Monday-Sunday.
# ============================================================

def scanner_loop():

    while True:

        try:

            next_scan = get_next_scan_time()

            with lock:

                if STATE["status"] != "SCANNING":
                    STATE["status"] = "WAITING FOR DAILY SCAN"

            logging.info(
                "Next automatic daily scan: %s",
                next_scan.isoformat()
            )

            while True:

                now = datetime.now(NY)

                seconds_remaining = (
                    next_scan - now
                ).total_seconds()

                if seconds_remaining <= 0:
                    break

                time.sleep(
                    min(
                        60,
                        max(
                            1,
                            seconds_remaining
                        )
                    )
                )

            logging.info(
                "Starting scheduled daily market scan..."
            )

            run_full_market_scan()

            # Prevent accidental duplicate run.
            time.sleep(90)

        except Exception:

            logging.exception(
                "Scanner loop error"
            )

            time.sleep(60)