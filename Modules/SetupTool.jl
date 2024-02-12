# Does not work for some reason

module SetupTool
    using Pkg

    export setupPackages

    function setupPackages(force = false, requiredPackages = ["GR", "Yao", "YaoPlots", "Plots", "BitBasis", "StatsBase", "Suppressor", "Revise"], packagesToBuild = ["GR"]; update_registry::Bool = true)
        if force || !(checkInstallation(requiredPackages)) || !is_environment_prepared()
            Pkg.activate(mktempdir())

            if update_registry
                Pkg.Registry.update()
            end
            
            for req in requiredPackages
                Pkg.add(req)
                if req in packagesToBuild
                    Pkg.build(req)
                end
            end

            set_environment_flag()
            return true
        end

        return false
    end

    function checkInstallation(required)
        for req in required
            if !(req in keys(Pkg.project().dependencies))
                return false
            end
        end

        return true
    end

    # OLD CODE
    #=begin
	using Pkg
	Pkg.activate(mktempdir())
	Pkg.Registry.update()
	Pkg.add("GR")
	Pkg.build("GR")
	Pkg.add("Yao")
	Pkg.add("YaoPlots")
	Pkg.add("Plots")
	Pkg.add("BitBasis")
	Pkg.add("StatsBase")
    end=#

    # Function to set a flag
    function set_environment_flag()
        # Define the flag file path
        flag_file = "env_prepared.flag"
        
        # Check if the flag file exists
        if !isfile(flag_file)
            # If it doesn't exist, create the flag file to indicate the environment is prepared
            open(flag_file, "w") do f
                write(f, "The environment is prepared.")
            end
            @info "Environment flag set."
        else
            @info "Environment has been prepared."
        end
    end

    # Function to check if the environment is prepared
    function is_environment_prepared()
        # Define the flag file path
        flag_file = "env_prepared.flag"
        
        # Check if the flag file exists
        return isfile(flag_file)
    end
end