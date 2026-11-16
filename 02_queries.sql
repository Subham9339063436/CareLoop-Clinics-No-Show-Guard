-- CareLoop Clinics — No-Show Guard
-- Part 2: SQL verification and analysis.
-- MySQL Workbench: run CREATE DATABASE / USE below.
-- SQLiteOnline: skip only those two lines; it accepts the remaining DDL and queries.

CREATE DATABASE noshowguard;
USE noshowguard;

CREATE TABLE clinics (
  clinic_id VARCHAR(5) PRIMARY KEY,
  clinic_name VARCHAR(40) NOT NULL,
  city VARCHAR(20) NOT NULL,
  tier VARCHAR(10),
  latitude DECIMAL(8,4),
  longitude DECIMAL(8,4),
  temp_c DECIMAL(5,1)
);

CREATE TABLE appointments (
  appointment_id VARCHAR(8) PRIMARY KEY,
  patient_id VARCHAR(8) NOT NULL,
  clinic_id VARCHAR(5) NOT NULL,
  booked_on DATE,
  appt_date DATE,
  age INT,
  gender VARCHAR(1),
  distance_km DECIMAL(5,1),
  fee_inr INT,
  booking_channel VARCHAR(10),
  visit_type VARCHAR(12),
  reminder_sent VARCHAR(3),
  prior_no_shows INT,
  status VARCHAR(10),
  lead_days INT,
  no_show INT,
  FOREIGN KEY (clinic_id) REFERENCES clinics(clinic_id)
);

-- Task 2.1: Run load_data.sql after creating both tables.
SELECT COUNT(*) AS clinic_count FROM clinics;
SELECT COUNT(*) AS appointment_count, SUM(no_show) AS no_shows FROM appointments;

-- Task 2.1: Duplicate-key test. This statement is expected to fail after data is loaded.
-- INSERT INTO appointments (appointment_id, patient_id, clinic_id) VALUES ('A0001','P196','C05');
-- Observed in SQLite (the permitted SQLiteOnline alternative):
-- UNIQUE constraint failed: appointments.appointment_id
-- MySQL Workbench reports a duplicate-entry / PRIMARY KEY error instead.

-- Task 2.2: No-show rate by clinic, retaining clinics with at least 95 appointments.
SELECT c.clinic_name,
       c.city,
       COUNT(*) AS appointments,
       SUM(a.no_show) AS no_shows,
       ROUND(100.0 * SUM(a.no_show) / COUNT(*), 1) AS no_show_pct
FROM appointments AS a
JOIN clinics AS c ON c.clinic_id = a.clinic_id
GROUP BY c.clinic_id, c.clinic_name, c.city
HAVING COUNT(*) >= 95
ORDER BY no_show_pct DESC;

-- Task 2.2: Repeat without HAVING to inspect all eight clinics.
SELECT c.clinic_name,
       c.city,
       COUNT(*) AS appointments,
       SUM(a.no_show) AS no_shows,
       ROUND(100.0 * SUM(a.no_show) / COUNT(*), 1) AS no_show_pct
FROM appointments AS a
JOIN clinics AS c ON c.clinic_id = a.clinic_id
GROUP BY c.clinic_id, c.clinic_name, c.city
ORDER BY no_show_pct DESC;

-- Python reconciliation used the clean CSV:
-- (appointments_clean.groupby('clinic_id')
--  .agg(appointments=('appointment_id','count'), no_shows=('no_show','sum')))
-- The clinic counts and no-show counts match the SQL results exactly.

-- Task 2.3: Lead-time bands. ORDER BY MIN(lead_days) ensures numeric band order.
SELECT CASE
         WHEN lead_days <= 7 THEN '1-7'
         WHEN lead_days <= 14 THEN '8-14'
         WHEN lead_days <= 30 THEN '15-30'
         ELSE '31+'
       END AS lead_band,
       COUNT(*) AS appointments,
       SUM(no_show) AS no_shows,
       ROUND(100.0 * SUM(no_show) / COUNT(*), 1) AS no_show_pct
FROM appointments
GROUP BY lead_band
ORDER BY MIN(lead_days);

-- Task 2.4: Month-by-month trend and change in percentage points.
WITH monthly AS (
  SELECT SUBSTR(appt_date, 1, 7) AS month,
         COUNT(*) AS appointments,
         ROUND(100.0 * SUM(no_show) / COUNT(*), 1) AS no_show_pct
  FROM appointments
  GROUP BY SUBSTR(appt_date, 1, 7)
)
SELECT month,
       appointments,
       no_show_pct,
       ROUND(no_show_pct - LAG(no_show_pct) OVER (ORDER BY month), 1) AS change_pp
FROM monthly
ORDER BY month;

-- Task 2.5: Rank clinics within their city and compare each rate with the overall rate.
WITH clinic_rates AS (
  SELECT c.clinic_name AS clinic,
         c.city,
         100.0 * SUM(a.no_show) / COUNT(*) AS no_show_pct
  FROM appointments AS a
  JOIN clinics AS c ON c.clinic_id = a.clinic_id
  GROUP BY c.clinic_id, c.clinic_name, c.city
)
SELECT clinic,
       city,
       ROUND(no_show_pct, 1) AS no_show_pct,
       RANK() OVER (PARTITION BY city ORDER BY no_show_pct DESC) AS rank_in_city,
       CASE
         WHEN no_show_pct > (SELECT 100.0 * SUM(no_show) / COUNT(*) FROM appointments) THEN 'above'
         WHEN no_show_pct < (SELECT 100.0 * SUM(no_show) / COUNT(*) FROM appointments) THEN 'below'
         ELSE 'equal'
       END AS vs_overall
FROM clinic_rates
ORDER BY city, rank_in_city;
