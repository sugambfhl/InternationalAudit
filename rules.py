import functools

import pandas as pd
from loguru import logger


def rule_method(active: bool = True):
    """
    Decorator factory.
    Use as: @rule_method(active=True)  # included
            @rule_method(active=False) # excluded
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # logger.info(f"Running: {func.__name__}")
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                df = None
                if len(args) >= 2:  # method
                    df = args[1]
                elif len(args) >= 1:  # plain function
                    df = args[0]
                return df

        wrapper._is_rule_method = True
        wrapper._rule_active = active
        return wrapper

    return decorator


class ComputeRule:
    def _check_extra_condition(
        self, df: pd.DataFrame, extra_condition: list[dict]
    ) -> pd.Series:
        mask = pd.Series([True] * len(df))

        for condition in extra_condition:
            col: str = condition.get("column", "")
            conds: dict = condition.get("condition", {})
            for op, val in conds.items():
                if op == "gte" and isinstance(val, (int, float)):
                    mask &= df[col] >= val
                elif op == "lte" and isinstance(val, (int, float)):
                    mask &= df[col] <= val
                elif op == "gt" and isinstance(val, (int, float)):
                    mask &= df[col] > val
                elif op == "lt" and isinstance(val, (int, float)):
                    mask &= df[col] < val
                elif op == "eq":
                    mask &= df[col].str == val
                elif op == "neq":
                    mask &= df[col].str != val
                elif op == "isin" and isinstance(val, list):
                    mask &= df[col].isin(val)
                elif op == "notin" and isinstance(val, list):
                    mask &= ~df[col].isin(val)
                else:
                    logger.warning(f"Invalid operation detected: {op}")
                    mask &= False
        return mask

    def _compute_inclusion_exclusion(
        self,
        df: pd.DataFrame,
        trigger_name: str,
        inclusion: list[str] | None = None,
        exclusion: list[str] | None = None,
        inclusion_column: str | None = None,
        exclusion_column: str | None = None,
        extra_condition: list[dict] | None = None,
    ):
        is_inclusion_present = pd.Series([True] * len(df))
        is_exclusion_absent = pd.Series([True] * len(df))
        is_extra_conditions_present = pd.Series([True] * len(df))

        if inclusion is None and exclusion is None and extra_condition is None:
            raise RuntimeError(
                "Inclusion, Exclusion and Extra Condition can not be None at the same time."
            )

        if inclusion:
            if inclusion_column not in df.columns:
                logger.warning(f"{inclusion_column} not present.")
                return df

            is_inclusion_present = df[inclusion_column].map(
                lambda x: any(code == x for code in inclusion)
            )
        if extra_condition:
            is_extra_conditions_present = self._check_extra_condition(
                df=df,
                extra_condition=extra_condition,
            )

        if exclusion:
            if not exclusion_column:
                logger.warning("exclusion_column is None but exclusion codes provided")
                return df

            if exclusion_column not in df.columns:
                logger.warning(f"{exclusion_column} not present in dataframe")
                return df

            is_exclusion_absent = df[exclusion_column].apply(
                lambda x: all(code != x for code in exclusion)
            )
        is_trigger_present = (
            is_inclusion_present & is_exclusion_absent & is_extra_conditions_present
        )
        df.loc[is_trigger_present, "Filter Applied"] = df.loc[
            is_trigger_present, "Filter Applied"
        ].apply(lambda x: x + [trigger_name])
        logger.success(f"Successfull Run: {trigger_name}")
        return df

    # def apply_all_rules(self, df):
    #     df = df.copy(deep=True)
    #     df = self.general_exclusion_hiv(df=df)
    #     df = self.general_exclusion_zirconium_crown(df=df)
    #     df = self.covid(df=df)
    #     df = self.hpv_screening(df=df)
    #     df = self.alopecia(df=df)
    #     df = self.more_than_one_quantity(df=df)
    #     df = self.sick_leave(df=df)
    #     df = self.pap_smear_age_restriction(df=df)
    #     df = self.desensitization(df=df)
    #     df = self.zinc_general_exclusion(df=df)
    #     df = self.betadine_mouth_wash(df=df)
    #     df = self.cough_syrup_high_quantity(df=df)
    #     df = self.nasal_syrup_high_quantity(df=df)
    #     df = self.nebulizer_high_quantity(df=df)
    #     df = self.hpyrol_antibody(df=df)
    #     df = self.gardenia_large_dressing(df=df)
    #     df = self.sidra_medical_male(df=df)
    #     return df
    def apply_all_rules(self, df):
        for name in dir(self):
            method = getattr(self, name)
            if callable(method) and getattr(method, "_is_rule_method", False):
                if getattr(method, "_rule_active", True):
                    df = method(df)
        return df

    @rule_method(active=True)
    def general_exclusion_hiv(self, df):
        inclusion = ["86689", "86701", "86702"]
        exclusion = ["OUT-PATIENT MATERNITY"]
        trigger_name = "General exclusion - HIV"
        df = self._compute_inclusion_exclusion(
            inclusion=inclusion,
            df=df,
            trigger_name=trigger_name,
            exclusion=exclusion,
            inclusion_column="ACTIVITY_CODE",
            exclusion_column="BENEFIT_TYPE",
        )
        return df

    @rule_method(active=True)
    def general_exclusion_zirconium_crown(self, df):
        trigger_name = "General exclusion-Zirconium Crown"
        inclusion = ["D2720", "D2750"]
        exclusion = [
            "AK/HC/00093/5/1",
            "AK/HC/00093/5/2",
            "AK/HC/00093/5/3",
            "AK/HC/00093/5/4",
            "AK/HC/00093/5/5",
            "AK/HC/00093/5/6",
            "AK/HC/00093/5/7",
            "AK/HC/00143/1/1",
            "AK/HC/00143/0/1",
            "AK/HC/00143/2/1",
            "AK/HC/00153/0/1",
            "AK/HC/00153/1/1",
        ]
        df = self._compute_inclusion_exclusion(
            inclusion=inclusion,
            exclusion=exclusion,
            df=df,
            trigger_name=trigger_name,
            inclusion_column="ACTIVITY_CODE",
            exclusion_column="POLICY_NUMBER",
        )
        return df

    @rule_method(active=True)
    def covid(self, df):
        icd_code = [
            "U07.1",
            "U09.9",
            "Z11.52",
            "Z20.822",
            "Z28.310",
            "Z28.311",
            "Z86.16",
        ]
        exclusion = [
            "AK/HC/00093/5/1",
            "AK/HC/00093/5/2",
            "AK/HC/00093/5/3",
            "AK/HC/00093/5/4",
            "AK/HC/00093/5/5",
            "AK/HC/00093/5/6",
            "AK/HC/00093/5/7",
        ]
        trigger_name = "General exclusion-COVID"
        df = self._compute_inclusion_exclusion(
            inclusion=icd_code,
            exclusion=exclusion,
            df=df,
            trigger_name=trigger_name,
            inclusion_column="PRIMARY_ICD_CODE",
            exclusion_column="POLICY_NUMBER",
        )
        return df

    @rule_method(active=True)
    def hpv_screening(self, df):
        inclusion = ["0096U", "0500T", "0429U", "87623", "87624", "87625", "0354U"]
        trigger_name = "General exclusion-HPV SCREENING"
        inclusion_column = "ACTIVITY_CODE"
        df = self._compute_inclusion_exclusion(
            inclusion=inclusion,
            df=df,
            trigger_name=trigger_name,
            inclusion_column=inclusion_column,
        )
        return df

    @rule_method(active=True)
    def alopecia(self, df):
        icd_inclusion = [
            "A51.32",
            "L63.0",
            "L63.1",
            "L63.8",
            "L63.9",
            "L64.0",
            "L64.8",
            "L64.9",
            "L65.2",
            "L66.8",
            "L66.9",
            "Q84.0",
            "L66.12",
            "L66.81",
            "L66.89",
        ]
        trigger_name = "General exclusion-ALOPECIA"
        df = self._compute_inclusion_exclusion(
            inclusion=icd_inclusion,
            df=df,
            trigger_name=trigger_name,
            inclusion_column="PRIMARY_ICD_CODE",
        )
        return df

    @rule_method(active=True)
    def more_than_one_quantity(self, df):
        inclusion = [
            "99202",
            "99203",
            "99204",
            "99205",
            "99211",
            "99212",
            "99213",
            "99214",
            "99215",
            "99221",
            "99222",
            "99223",
            "99231",
            "99232",
            "99233",
            "99234",
            "99235",
            "99236",
            "99238",
            "99239",
            "99242",
            "99243",
            "99244",
            "99245",
            "99252",
            "99253",
            "99254",
            "99255",
            "99281",
            "99282",
            "99283",
            "99284",
            "99285",
            "99288",
            "99291",
            "99292",
            "99304",
            "99305",
            "99306",
            "99307",
            "99308",
            "99309",
            "99310",
            "99315",
            "99316",
            "99341",
            "99342",
            "99344",
            "99345",
            "99347",
            "99348",
            "99349",
            "99350",
            "99358",
            "99359",
            "99360",
            "99366",
            "99367",
            "99368",
            "99374",
            "99375",
            "99377",
            "99378",
            "99379",
            "99380",
            "99381",
            "99382",
            "99383",
            "99384",
            "99385",
            "99386",
            "99387",
            "99391",
            "99392",
            "99393",
            "99394",
            "99395",
            "99396",
            "99397",
            "99401",
            "99402",
            "99403",
            "99404",
            "99406",
            "99407",
            "99408",
            "99409",
            "99411",
            "99412",
            "99429",
            "99441",
            "99442",
            "99443",
            "99450",
            "99455",
            "99456",
            "99460",
            "99461",
            "99462",
            "99463",
            "99464",
            "99465",
            "99466",
            "99467",
            "99468",
            "99469",
            "99471",
            "99472",
            "99475",
            "99476",
            "99477",
            "99478",
            "99479",
            "99480",
            "99499",
            "99500",
            "99501",
            "99502",
            "99503",
            "99504",
            "99505",
            "99506",
            "99507",
            "99509",
            "99510",
            "99511",
            "99512",
            "99600",
            "99601",
            "99602",
            "99605",
            "99606",
            "99607",
            "10",
            "61.08",
            "D9310",
            "61.11",
            "10.01",
            "9",
            "63",
            "11.01",
            "11",
            "99242",
            "99241",
            "61.03",
            "99253",
            "99243",
            "10.02",
            "22",
            "D0160",
            "88321",
            "21",
            "61.04",
            "61.01",
            "61.06",
            "61.02",
            "61.07",
            "61.09",
            "61.12",
            "63.01",
            "63.02",
            "63.03",
            "63.04",
            "63.05",
            "23",
            "61.05",
            "9.01",
            "9.02",
            "11.02",
            "13",
            "70450",
            "70460",
            "70470",
            "70480",
            "70481",
            "70482",
            "70486",
            "70487",
            "70488",
            "70490",
            "70491",
            "70492",
            "71250",
            "71260",
            "71270",
            "72125",
            "72126",
            "72127",
            "72128",
            "72129",
            "72130",
            "72131",
            "72132",
            "72133",
            "74150",
            "74160",
            "74170",
            "74176",
            "74177",
            "74178",
            "72191",
            "72192",
            "72193",
            "70496",
            "70498",
            "71275",
            "73706",
            "74174",
            "70551",
            "70552",
            "70553",
            "70540",
            "70542",
            "70543",
            "72141",
            "72142",
            "72156",
            "72146",
            "72147",
            "72157",
            "72148",
            "72149",
            "72158",
            "73218",
            "73219",
            "73220",
            "73721",
            "73722",
            "73723",
            "74181",
            "74182",
            "74183",
            "72195",
            "72196",
            "72197",
            "75557",
            "75561",
            "77046",
            "77047",
            "77048",
            "77049",
            "71271",
            "74712",
            "74713",
            "75580",
            "76391",
            "70544",
            "70545",
            "70546",
            "70547",
            "70548",
            "70549",
            "70554",
            "72194",
            "72198",
            "73700",
            "73701",
            "73702",
            "73718",
            "73719",
            "74185",
            "75559",
            "75563",
            "77011",
            "77012",
            "77013",
            "77014",
            "77021",
            "77022",
        ]
        extra_conditions: list[dict] = [
            {"column": "ACTIVITY_QUANTITY_APPROVED", "condition": {"gt": 1}}
        ]
        trigger_name = "Quantity More Than 1"
        df = self._compute_inclusion_exclusion(
            inclusion=inclusion,
            df=df,
            trigger_name=trigger_name,
            inclusion_column="ACTIVITY_CODE",
            extra_condition=extra_conditions,
        )
        return df

    @rule_method(active=True)
    def sick_leave(self, df):
        if "PRESENTING_COMPLAINTS" not in df.columns:
            logger.error("Presenting Complainst not in data.")
            return df

        is_sick_present = (
            df["PRESENTING_COMPLAINTS"].str.lower().str.contains("sick").fillna(False)
        )
        trigger_name = "General exclusion - Sick Leave"
        df.loc[is_sick_present, "Filter Applied"] = df.loc[
            is_sick_present, "Filter Applied"
        ].apply(lambda x: x + [trigger_name])
        return df

    @rule_method(active=True)
    def pap_smear_age_restriction(self, df):
        trigger_name: str = "PAP Smear Age Restriction"
        inclusion = [
            "88141",
            "88142",
            "88143",
            "88147",
            "88148",
            "88150",
            "88152",
            "88153",
            "88155",
            "88164",
            "88165",
            "88166",
            "88167",
            "88174",
            "88175",
            "88177",
        ]
        exclusion: list[str] = [
            "AL EMADI HOSPITAL",
            "AL EMADI HOSPITAL CLINICS - NORTH",
        ]
        inclusion_column: str = "ACTIVITY_CODE"
        exclusion_column: str = "PROVIDER_NAME"

        extra_conditions: list[dict] = [
            {"column": "MEMBER_AGE", "condition": {"lte": 24, "gte": 64}}
        ]

        df = self._compute_inclusion_exclusion(
            inclusion=inclusion,
            exclusion=exclusion,
            df=df,
            trigger_name=trigger_name,
            inclusion_column=inclusion_column,
            exclusion_column=exclusion_column,
            extra_condition=extra_conditions,
        )
        return df

    @rule_method(active=True)
    def desensitization(self, df):
        trigger_name: str = "Desensitization"
        inclusion: list[str] = ["D9910"]
        extra_conditions: list[dict] = [
            {"column": "MEMBER_AGE", "condition": {"gt": 18}}
        ]
        df = self._compute_inclusion_exclusion(
            df=df,
            trigger_name=trigger_name,
            inclusion=inclusion,
            inclusion_column="ACTIVITY_CODE",
            extra_condition=extra_conditions,
        )
        return df

    @rule_method(active=True)
    def zinc_general_exclusion(self, df):
        trigger_name: str = "Zinc-General Exclusion"
        inclusion: list[str] = ["84630"]
        exclusion = ["HEALTH CHECK-UP"]
        inclusion_column = "ACTIVITY_CODE"
        exclusion_column = "BENEFIT_TYPE"
        df = self._compute_inclusion_exclusion(
            df=df,
            trigger_name=trigger_name,
            inclusion=inclusion,
            exclusion=exclusion,
            inclusion_column=inclusion_column,
            exclusion_column=exclusion_column,
        )
        return df

    @rule_method(active=True)
    def betadine_mouth_wash(self, df):
        trigger_name: str = "Betadine Mouth wash"
        inclusion: list[str] = ["0000-000000-001427"]
        exclusion: list[str] = ["AK/HC/00156/0/1"]
        exclusion_column: str = "POLICY_NUMBER"
        inclusion_column: str = "ACTIVITY_CODE"
        df = self._compute_inclusion_exclusion(
            df=df,
            trigger_name=trigger_name,
            inclusion=inclusion,
            exclusion=exclusion,
            inclusion_column=inclusion_column,
            exclusion_column=exclusion_column,
        )
        return df

    @rule_method(active=True)
    def cough_syrup_high_quantity(self, df):
        trigger_name: str = "Cough Syrup-Quantity 2"
        extra_conditions: list[dict] = [
            {
                "column": "ACTIVITY_INTERNAL_DESCRIPTION",
                "condition": {
                    "isin": [
                        "PROSPAN COUGH SYRUP",
                        "LINKUS COUGH SYRUP",
                        "THIVY COUGH SYRUP",
                        "ZECUF SYRUP SUGER FREE",
                        "STOPKOF SYRUP",
                        "IVY COUGH SYRUP",
                        "MUNATORIL COLD COUGH SYRUP",
                        "Dextrokuf Syrup",
                        "EZE IVY SYRUP",
                        "LIBLAB SYRUP",
                    ]
                },
            },
            {"column": "ACTIVITY_QUANTITY_APPROVED", "condition": {"gt": 2}},
        ]
        df = self._compute_inclusion_exclusion(
            df=df, trigger_name=trigger_name, extra_condition=extra_conditions
        )
        return df

    @rule_method(active=True)
    def nasal_syrup_high_quantity(self, df):
        trigger_name: str = "Nasal Spray-Quantity 2"
        extra_conditions: list[dict] = [
            {
                "column": "ACTIVITY_INTERNAL_DESCRIPTION",
                "condition": {"isin": ["NASAL SPRAY"]},
            },
            {"column": "ACTIVITY_QUANTITY_APPROVED", "condition": {"gt": 2}},
        ]
        df = self._compute_inclusion_exclusion(
            df=df, trigger_name=trigger_name, extra_condition=extra_conditions
        )
        return df

    @rule_method(active=True)
    def nebulizer_high_quantity(self, df):
        trigger_name: str = "Nebulizer- Quantity 1"
        inclusion: list[str] = ["94640"]
        inclusion_column: str = "ACTIVITY_CODE"
        extra_conditions: list[dict] = [
            {"column": "ACTIVITY_QUANTITY_APPROVED", "condition": {"gt": 1}},
        ]
        df = self._compute_inclusion_exclusion(
            df=df,
            trigger_name=trigger_name,
            inclusion=inclusion,
            inclusion_column=inclusion_column,
            extra_condition=extra_conditions,
        )
        return df

    @rule_method(active=True)
    def hpyrol_antibody(self, df):
        trigger_name: str = "H-Pylori Antibody not covered"
        inclusion: list[str] = ["86677"]
        inclusion_column: str = "ACTIVITY_CODE"
        df = self._compute_inclusion_exclusion(
            df=df,
            trigger_name=trigger_name,
            inclusion=inclusion,
            inclusion_column=inclusion_column,
        )
        return df

    @rule_method(active=True)
    def gardenia_large_dressing(self, df):
        trigger_name: str = "Gardenia-Large Dressing not covered"
        extra_conditions: list[dict] = [
            {
                "column": "ACTIVITY_INTERNAL_DESCRIPTION",
                "condition": {"isin": ["Large Dressing", "Large Dressing"]},
            },
            {
                "column": "PROVIDER_NAME",
                "condition": {"eq": "GARDENIA MEDICAL CENTER"},
            },
        ]
        df = self._compute_inclusion_exclusion(
            df=df,
            trigger_name=trigger_name,
            extra_condition=extra_conditions,
        )
        return df

    @rule_method(active=True)
    def sidra_medical_male(self, df):
        trigger_name: str = "Sidra Medical Male Above 17 Years"
        extra_conditions: list[dict] = [
            {
                "column": "PROVIDER_NAME",
                "condition": {"eq": "SIDRA MEDICAL AND RESEARCH CENTER"},
            },
            {
                "column": "MEMBER_AGE",
                "condition": {"gt": 17},
            },
            {
                "column": "GENDER",
                "condition": {"eq": "Male"},
            },
        ]
        df = self._compute_inclusion_exclusion(
            df=df,
            trigger_name=trigger_name,
            extra_condition=extra_conditions,
        )
        return df
