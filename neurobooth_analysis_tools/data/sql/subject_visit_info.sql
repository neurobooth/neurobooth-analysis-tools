WITH vd_dist AS (
    -- Figure out which demographic dates before the visit dates are closest (distance = 1)
    SELECT
        visit.subject_id,
        visit.neurobooth_visit_dates AS visit_date,
        demog.end_time_demographic AS demog_date,
        visit.neurobooth_visit_dates - DATE(demog.end_time_demographic) AS offset_days,
        ROW_NUMBER() OVER (
                PARTITION BY visit.subject_id, visit.neurobooth_visit_dates
                ORDER BY ABS(visit.neurobooth_visit_dates - DATE(demog.end_time_demographic)) ASC
        ) AS distance
    FROM rc_visit_dates visit
    FULL OUTER JOIN rc_demographic demog
        ON visit.subject_id = demog.subject_id
    WHERE visit.neurobooth_visit_dates IS NOT NULL
), vc_dist AS (
    -- Figure out which clinic dates before the visit dates are closest (distance = 1)
    SELECT
        visit.subject_id,
        visit.neurobooth_visit_dates AS visit_date,
        clin.date_enrolled AS clin_date,
        visit.neurobooth_visit_dates - clin.date_enrolled AS offset_days,
        ROW_NUMBER() OVER (
                PARTITION BY visit.subject_id, visit.neurobooth_visit_dates
                ORDER BY ABS(visit.neurobooth_visit_dates - clin.date_enrolled) ASC
        ) AS distance
    FROM rc_visit_dates visit
    FULL OUTER JOIN rc_clinical clin
        ON visit.subject_id = clin.subject_id
    WHERE visit.neurobooth_visit_dates IS NOT NULL
), vs_dist AS (
    -- Figure out which scale dates before the visit dates are closest (distance = 1)
        SELECT
            visit.subject_id,
            visit.neurobooth_visit_dates AS visit_date,
            scales.end_time_ataxia_pd_scales AS scale_date,
            visit.neurobooth_visit_dates - DATE(scales.end_time_ataxia_pd_scales) AS offset_days,
            ROW_NUMBER() OVER (
                    PARTITION BY visit.subject_id, visit.neurobooth_visit_dates
                    ORDER BY ABS(visit.neurobooth_visit_dates - DATE(scales.end_time_ataxia_pd_scales)) ASC
            ) AS distance
        FROM rc_visit_dates visit
        FULL OUTER JOIN rc_ataxia_pd_scales scales
            ON visit.subject_id = scales.subject_id
        WHERE visit.neurobooth_visit_dates IS NOT NULL
)
SELECT
    subj.subject_id,
    consent.test_subject_boolean,
    vd_dist.visit_date,
    DATE(vd_dist.demog_date) AS demog_date,
    vd_dist.offset_days AS demog_offset_days,
    demog.end_time_demographic IS NOT NULL AND ABS(vd_dist.offset_days) <= 60 AS recent_demog,
    vc_dist.clin_date,
    vc_dist.offset_days AS clin_offset_days,
    clin.date_enrolled IS NOT NULL AND ABS(vc_dist.offset_days) <= 60 AS recent_clin,
    DATE(vs_dist.scale_date) AS scale_date,
    vs_dist.offset_days AS scales_offset_days,
    scales.end_time_ataxia_pd_scales IS NOT NULL AND ABS(vs_dist.offset_days) <= 60 AS recent_scale,
    EXTRACT(YEAR FROM AGE(vd_dist.visit_date, subj.date_of_birth_subject)) AS age,
    CASE
        WHEN CAST(CAST(subj.gender_at_birth AS FLOAT) AS INT) = 1 THEN 'Male'
        WHEN CAST(CAST(subj.gender_at_birth AS FLOAT) AS INT) = 2 THEN 'Female'
        ELSE '?'
    END AS gender_birth,
    CASE
        WHEN demog.handedness = 0 THEN 'Right'
        WHEN demog.handedness = 1 THEN 'Left'
        WHEN demog.handedness = 2 THEN 'Ambi'
        ELSE '?'
    END AS handedness,
    clin.primary_diagnosis,
    CASE  -- Case statement also serves to return false for NULL entries
        WHEN 0 = ANY(clin.primary_diagnosis) THEN TRUE
        ELSE FALSE
    END AS is_control,
    CASE
        WHEN scales.bars_gait > 95 THEN NULL
        ELSE CAST(scales.bars_gait AS FLOAT) / 10
    END AS bars_gait,
    CASE
        WHEN scales.bars_heel_shin_right > 95 THEN NULL
        ELSE CAST(scales.bars_heel_shin_right AS FLOAT) / 10
    END AS bars_heel_shin_right,
    CASE
        WHEN scales.bars_heel_shin_left > 95 THEN NULL
        ELSE CAST(scales.bars_heel_shin_left AS FLOAT) / 10
    END AS bars_heel_shin_left,
    CASE
        WHEN scales.bars_finger_nose_right > 95 THEN NULL
        ELSE CAST(scales.bars_finger_nose_right AS FLOAT) / 10
    END AS bars_finger_nose_right,
    CASE
        WHEN scales.bars_finger_nose_left > 95 THEN NULL
        ELSE CAST(scales.bars_finger_nose_left AS FLOAT) / 10
    END AS bars_finger_nose_left,
    CASE
        WHEN scales.bars_speech > 95 THEN NULL
        ELSE CAST(scales.bars_speech AS FLOAT) / 10
    END AS bars_speech,
    CASE
        WHEN scales.bars_oculomotor > 95 THEN NULL
        ELSE CAST(scales.bars_oculomotor AS FLOAT) / 10
    END AS bars_oculomotor
FROM subject subj
LEFT JOIN rc_participant_and_consent_information consent
	ON subj.subject_id = consent.subject_id
LEFT JOIN vd_dist
    ON subj.subject_id = vd_dist.subject_id
    AND vd_dist.distance = 1
LEFT JOIN vc_dist
    ON subj.subject_id = vc_dist.subject_id
    AND vc_dist.distance = 1
LEFT JOIN vs_dist
    ON subj.subject_id = vs_dist.subject_id
    AND vs_dist.distance = 1
LEFT JOIN rc_visit_dates visit
    ON subj.subject_id = visit.subject_id
    AND vd_dist.visit_date = visit.neurobooth_visit_dates
    AND vc_dist.visit_date = visit.neurobooth_visit_dates
    AND vs_dist.visit_date = visit.neurobooth_visit_dates
LEFT JOIN rc_demographic demog
    ON subj.subject_id = demog.subject_id
    AND vd_dist.demog_date = demog.end_time_demographic
LEFT JOIN rc_clinical clin
    ON subj.subject_id = clin.subject_id
    AND vc_dist.clin_date = clin.date_enrolled
LEFT JOIN rc_ataxia_pd_scales scales
    ON subj.subject_id = scales.subject_id
    AND vs_dist.scale_date = scales.end_time_ataxia_pd_scales
WHERE visit.neurobooth_visit_dates IS NOT NULL
ORDER BY
    CAST(subj.subject_id AS INT) ASC,
    visit.neurobooth_visit_dates ASC
