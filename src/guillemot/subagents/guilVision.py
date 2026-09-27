from dotenv import load_dotenv

load_dotenv()


import os

from pydantic_ai import Agent
from pydantic import BaseModel, Field
from guillemot.model import build_model, build_model_settings
from guillemot.model import context_limit 


class XRDPeak(BaseModel):
    two_theta: float = Field(description="Peak position in 2θ (degrees)")
    intensity: float = Field(description="Peak intensity")

class XRDAnalysisResult(BaseModel):
    peaks: list[XRDPeak]
    background_noise_level: str = Field(description="High, medium, or low background noise level")
    phase_identified: str = Field(description="Likely phase(s) in the sample", default=None)
    crystallinity : str = Field(description="Crystalline, semicrystalline, or amorphous with reasoning")
    output: str = Field(description="Plain text summary of the output")

class FitQuality(BaseModel):
    observed_peaks: list[XRDPeak] = Field(description="The experimental/observed peak positions in 2θ (degrees) and intensities")
    calculated_peaks: list[XRDPeak] = Field(description="The calculated peak positions in 2θ (degrees) and intensities by TOPAS")
    risidual_of_peaks: float = Field(description="""Risidual of the observed and calculated peak intensities by the formula: Sy = Σ wᵢ (yᵢ,obs - yᵢ,calc)²
    where wᵢ = 1/yᵢ,obs (counting statistics weight), summed over all data points i. """)




# LLM model for guilVision
model_name= os.getenv("GUILVISION_AI_MODEL", "ollama:qwen3.6:27b")
model= build_model(model_name)
model_settings=  build_model_settings(model_name)
GUILVISION_CONTEXT_LIMIT= context_limit(model_name)

 
guilVision = Agent(
    model= model,
    name="guilVision",
    description="sub-agnet that analyzes powder X-ray Diffraction (XRD) images/plots and return the information in a structured text format. ",
    output_type= XRDAnalysisResult | FitQuality,
    model_settings=model_settings,
    retries= 3, 
    system_prompt= """
    You are a powder X-ray diffraction (PXRD) plot analyst.
    You are given an image of a PXRD plot (2-theta on x-axis, intensity on y-axis).

    Extract: 
    - approximate 2-theta positions and relative intensities of major peaks
    - background noise level
    - likely phase(s) present (if identifiable from the peaks)
    - crystallinity assessment (sharp peaks = crystalline, broad humps = amorphous)
    - a short plain-text summary 

    When given images of XRD plots after refinement by TOPAS, you will analyse the image 
    and report the 'fit quality' back to @guillemot so that it can plan for the next refinement.
 
    Do not invent precise values you cannot read from the plot. State
    approximate values and flag low confidence where the image is unclear.""",
)


if __name__ == "__main__":

    result = guilVision.run_sync(
        "Analyze this PXRD plot at ./example_xrd_plot.png",
    )
    print(result.output)